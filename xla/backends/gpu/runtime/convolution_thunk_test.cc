/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/runtime/convolution_thunk.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::NiceMock;
using ::testing::Return;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

template <typename Sig>
class FakeRunner;

template <typename... Args>
class FakeRunner<void(Args...)> : public se::dnn::OpRunner<void(Args...)> {
 public:
  explicit FakeRunner(int* launches) : launches_(launches) {}

  std::string ToString() const override { return "fake"; }
  size_t GetWorkspaceSize() const override { return 0; }
  absl::StatusOr<se::dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return se::dnn::AlgorithmDesc();
  }
  absl::Status operator()(se::Stream*, se::dnn::ProfileResult*,
                          se::DeviceAddressBase, Args...) const override {
    ++*launches_;
    return absl::OkStatus();
  }

 private:
  int* launches_;
};

class FakeDnnSupport : public se::dnn::DnnSupport {
 public:
  using se::dnn::DnnSupport::DoPoolBackward;
  using se::dnn::DnnSupport::DoPoolForward;

  int runner_requests() const { return runner_requests_; }
  int launches() const { return launches_; }

  absl::Status Init() override { return absl::OkStatus(); }

  absl::StatusOr<std::unique_ptr<const se::dnn::ConvRunner>>
  ConvolveRunnerFromDesc(se::Stream*, const se::dnn::AlgorithmDesc&,
                         se::dnn::ConvolutionKind, se::dnn::DataType,
                         se::dnn::DataType, const se::dnn::BatchDescriptor&,
                         const se::dnn::FilterDescriptor&,
                         const se::dnn::BatchDescriptor&,
                         const se::dnn::ConvolutionDescriptor&) override {
    return CreateRunner<se::dnn::ConvSignature>();
  }

  absl::StatusOr<std::unique_ptr<const se::dnn::GraphConvRunner>>
  GraphConvolveRunnerFromDesc(
      se::Stream*, const se::dnn::AlgorithmDesc&, se::dnn::ConvolutionKind,
      se::dnn::DataType, se::dnn::DataType, const se::dnn::BatchDescriptor&,
      const se::dnn::FilterDescriptor&, const se::dnn::BatchDescriptor&,
      const se::dnn::ConvolutionDescriptor&, std::string) override {
    return CreateRunner<se::dnn::GraphConvSignature>();
  }

  absl::StatusOr<std::unique_ptr<const se::dnn::FusedConvRunner>>
  FusedConvolveRunnerFromDesc(
      se::Stream*, const se::dnn::AlgorithmDesc&, se::dnn::ConvolutionKind,
      se::dnn::DataType, se::dnn::DataType, se::dnn::DataType, double, double,
      double, const se::dnn::BatchDescriptor&, const se::dnn::FilterDescriptor&,
      const se::dnn::BatchDescriptor&, const se::dnn::BatchDescriptor&,
      const se::dnn::ConvolutionDescriptor&, se::dnn::ActivationMode) override {
    return CreateRunner<se::dnn::FusedConvSignature>();
  }

  absl::Status DoPoolForward(se::dnn::DataType, se::Stream*,
                             const se::dnn::PoolingDescriptor&,
                             const se::dnn::BatchDescriptor&,
                             se::DeviceAddressBase,
                             const se::dnn::BatchDescriptor&,
                             se::DeviceAddressBase,
                             se::ScratchAllocator*) override {
    return absl::UnimplementedError("DoPoolForward");
  }

  absl::Status DoPoolBackward(se::dnn::DataType, se::Stream*,
                              const se::dnn::PoolingDescriptor&,
                              const se::dnn::BatchDescriptor&,
                              se::DeviceAddressBase,
                              const se::dnn::BatchDescriptor&,
                              se::DeviceAddressBase, se::DeviceAddressBase,
                              se::DeviceAddressBase,
                              se::ScratchAllocator*) override {
    return absl::UnimplementedError("DoPoolBackward");
  }

 private:
  template <typename Sig>
  absl::StatusOr<std::unique_ptr<const se::dnn::OpRunner<Sig>>> CreateRunner() {
    ++runner_requests_;
    return std::make_unique<FakeRunner<Sig>>(&launches_);
  }

  int runner_requests_ = 0;
  int launches_ = 0;
};

struct FakeDevice {
  explicit FakeDevice(int ordinal) {
    ON_CALL(executor, device_ordinal()).WillByDefault(Return(ordinal));
    ON_CALL(executor, AsDnn()).WillByDefault(Return(&dnn));
    ON_CALL(stream, parent()).WillByDefault(Return(&executor));
    ON_CALL(stream, ok()).WillByDefault(Return(true));
  }

  FakeDnnSupport dnn;
  NiceMock<se::MockStreamExecutor> executor;
  NiceMock<se::MockStream> stream;
};

class TestConvolution {
 public:
  static absl::StatusOr<std::unique_ptr<TestConvolution>> Create(
      CudnnConvKind kind, int devices_per_host) {
    auto conv = std::unique_ptr<TestConvolution>(new TestConvolution());
    Shape shape =
        ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 1, 1, 1}, {3, 2, 1, 0});

    GpuConvDescriptor descriptor;
    descriptor.kind = kind;
    descriptor.backend_config.set_conv_result_scale(1);
    descriptor.operand0_shape = shape;
    descriptor.operand1_shape = shape;
    descriptor.result_shape = shape;
    descriptor.scratch_size = sizeof(float);
    for (int i = 0; i < 2; ++i) {
      WindowDimension* dim = descriptor.window.add_dimensions();
      dim->set_size(1);
      dim->set_stride(1);
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
    }
    ConvolutionDimensionNumbers& dnums = descriptor.dnums;
    dnums.set_input_batch_dimension(0);
    dnums.set_input_feature_dimension(1);
    dnums.set_kernel_output_feature_dimension(0);
    dnums.set_kernel_input_feature_dimension(1);
    dnums.set_output_batch_dimension(0);
    dnums.set_output_feature_dimension(1);
    for (int64_t dim : {2, 3}) {
      dnums.add_input_spatial_dimensions(dim);
      dnums.add_kernel_spatial_dimensions(dim);
      dnums.add_output_spatial_dimensions(dim);
    }
    descriptor.feature_group_count = 1;

    auto slice = [&](int index) {
      return ShapedSlice{
          BufferAllocation::Slice(&conv->allocations_[index], 0, sizeof(float)),
          shape};
    };
    std::vector<ShapedSlice> operands = {slice(kInput), slice(kFilter)};
    if (kind == CudnnConvKind::kForwardActivation) {
      operands.push_back(slice(kBias));
    }
    ABSL_ASSIGN_OR_RETURN(
        conv->thunk_,
        ConvolutionThunk::Create(
            Thunk::ThunkInfo(), descriptor, operands, {slice(kResult)},
            BufferAllocation::Slice(&conv->allocations_[kScratch], 0,
                                    sizeof(float)),
            devices_per_host));
    return conv;
  }

  absl::Status Initialize(FakeDevice& device) {
    Thunk::InitializeParams params;
    params.executor = &device.executor;
    params.stream = &device.stream;
    return thunk_->Initialize(params);
  }

  absl::Status Execute(FakeDevice& device) {
    std::vector<se::DeviceAddressBase> buffers;
    for (float& buffer : memory_) {
      buffers.emplace_back(&buffer, sizeof(float));
    }
    BufferAllocations allocations(buffers, device.executor.device_ordinal(),
                                  /*memory_allocator=*/nullptr);
    ServiceExecutableRunOptions run_options;
    Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
        run_options, allocations, &device.stream, &device.stream,
        /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
        /*collective_memory=*/nullptr);
    return thunk_->ExecuteOnStream(params);
  }

 private:
  enum Buffer { kInput, kFilter, kBias, kResult, kScratch, kNumBuffers };

  TestConvolution() {
    allocations_.reserve(kNumBuffers);
    for (int i = 0; i < kNumBuffers; ++i) {
      allocations_.emplace_back(/*index=*/i, /*size=*/sizeof(float),
                                /*color=*/0);
    }
  }

  std::vector<BufferAllocation> allocations_;
  std::array<float, kNumBuffers> memory_ = {};
  std::unique_ptr<ConvolutionThunk> thunk_;
};

class ConvolutionThunkRunnerTest
    : public ::testing::TestWithParam<CudnnConvKind> {};

TEST_P(ConvolutionThunkRunnerTest, InitializeCreatesTheRunnerThatExecuteUses) {
  FakeDevice device(/*ordinal=*/0);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TestConvolution> conv,
      TestConvolution::Create(GetParam(), /*devices_per_host=*/1));

  ASSERT_OK(conv->Initialize(device));
  EXPECT_EQ(device.dnn.runner_requests(), 1);
  ASSERT_OK(conv->Initialize(device));
  EXPECT_EQ(device.dnn.runner_requests(), 1);

  // Under command buffers Execute is captured, so it must not create a runner.
  ASSERT_OK(conv->Execute(device));
  ASSERT_OK(conv->Execute(device));
  EXPECT_EQ(device.dnn.runner_requests(), 1);
  EXPECT_EQ(device.dnn.launches(), 2);
}

INSTANTIATE_TEST_SUITE_P(
    AllConvolutionKinds, ConvolutionThunkRunnerTest,
    ::testing::Values(CudnnConvKind::kForward, CudnnConvKind::kBackwardInput,
                      CudnnConvKind::kBackwardFilter,
                      CudnnConvKind::kForwardActivation,
                      CudnnConvKind::kForwardGraph),
    [](const ::testing::TestParamInfo<CudnnConvKind>& info) {
      return absl::StrReplaceAll(CudnnConvKindToString(info.param),
                                 {{" ", "_"}});
    });

TEST(ConvolutionThunkTest, ProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<ThunkProto>(R"pb(
    thunk_info { profile_annotation: "conv_thunk_profile" }
    convolution_thunk {
      conv_descriptor {
        kind: FORWARD
        backend_config {
          conv_result_scale: 1
          activation_mode: 0
          side_input_scale: 0
          leakyrelu_alpha: 0
        }
        window {
          dimensions {
            size: 1
            stride: 1
            padding_low: 0
            padding_high: 0
            window_dilation: 1
            base_dilation: 1
            window_reversal: false
          }
          dimensions {
            size: 1
            stride: 1
            padding_low: 0
            padding_high: 0
            window_dilation: 1
            base_dilation: 1
            window_reversal: false
          }
        }
        operand0_shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
        operand1_shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
        result_shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
        scratch_size: 1024
        dnums {
          input_batch_dimension: 0
          input_feature_dimension: 1
          input_spatial_dimensions: [ 2, 3 ]
          kernel_input_feature_dimension: 1
          kernel_output_feature_dimension: 0
          kernel_spatial_dimensions: [ 2, 3 ]
          output_batch_dimension: 0
          output_feature_dimension: 1
          output_spatial_dimensions: [ 2, 3 ]
        }
      }
      operand_buffers {
        slice { offset: 0 size: 4 buffer_allocation_index: 0 }
        shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
      }
      operand_buffers {
        slice { offset: 0 size: 4 buffer_allocation_index: 1 }
        shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
      }
      result_buffers {
        slice { offset: 0 size: 4 buffer_allocation_index: 2 }
        shape {
          element_type: F32
          dimensions: [ 1, 1, 1, 1 ]
          layout {
            minor_to_major: [ 3, 2, 1, 0 ]
            tail_padding_alignment_in_elements: 1
          }
          is_dynamic_dimension: [ false, false, false, false ]
        }
      }
      scratch_buffer { offset: 0 size: 1024 buffer_allocation_index: 3 }
    }
  )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/4, /*color=*/0);
  buffer_allocations.emplace_back(/*index=*/3, /*size=*/1024, /*color=*/0);

  ASSERT_OK_AND_ASSIGN(Thunk::ThunkInfo thunk_info,
                       Thunk::ThunkInfo::FromProto(proto.thunk_info()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ConvolutionThunk> thunk,
      ConvolutionThunk::FromProto(thunk_info, proto.convolution_thunk(),
                                  buffer_allocations,
                                  /*devices_per_host=*/1));
  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
