// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/face_detection/face_detection.proto

#include "mediapipe/modules/face_detection/face_detection.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace mediapipe {
constexpr FaceDetectionOptions::FaceDetectionOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : strides_()
  , model_path_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , delegate_(nullptr)
  , gpu_origin_(0)

  , tensor_width_(0)
  , tensor_height_(0)
  , num_layers_(0)
  , num_boxes_(0)
  , x_scale_(0)
  , y_scale_(0)
  , w_scale_(0)
  , h_scale_(0)
  , min_score_thresh_(0)
  , interpolated_scale_aspect_ratio_(1){}
struct FaceDetectionOptionsDefaultTypeInternal {
  constexpr FaceDetectionOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~FaceDetectionOptionsDefaultTypeInternal() {}
  union {
    FaceDetectionOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT FaceDetectionOptionsDefaultTypeInternal _FaceDetectionOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, model_path_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, gpu_origin_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, tensor_width_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, tensor_height_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, num_layers_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, strides_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, interpolated_scale_aspect_ratio_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, num_boxes_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, x_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, y_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, w_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, h_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, min_score_thresh_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FaceDetectionOptions, delegate_),
  0,
  2,
  3,
  4,
  5,
  ~0u,
  12,
  6,
  7,
  8,
  9,
  10,
  11,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 20, -1, sizeof(::mediapipe::FaceDetectionOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_FaceDetectionOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n5mediapipe/modules/face_detection/face_"
  "detection.proto\022\tmediapipe\0327mediapipe/ca"
  "lculators/tensor/inference_calculator.pr"
  "oto\032,mediapipe/framework/calculator_opti"
  "ons.proto\032\036mediapipe/gpu/gpu_origin.prot"
  "o\"\346\003\n\024FaceDetectionOptions\022\022\n\nmodel_path"
  "\030\001 \001(\t\022-\n\ngpu_origin\030\013 \001(\0162\031.mediapipe.G"
  "puOrigin.Mode\022\024\n\014tensor_width\030\025 \001(\005\022\025\n\rt"
  "ensor_height\030\026 \001(\005\022\022\n\nnum_layers\030\027 \001(\005\022\017"
  "\n\007strides\030\030 \003(\005\022*\n\037interpolated_scale_as"
  "pect_ratio\030\031 \001(\002:\0011\022\021\n\tnum_boxes\030\037 \001(\005\022\022"
  "\n\007x_scale\030  \001(\002:\0010\022\022\n\007y_scale\030! \001(\002:\0010\022\022"
  "\n\007w_scale\030\" \001(\002:\0010\022\022\n\007h_scale\030# \001(\002:\0010\022\030"
  "\n\020min_score_thresh\030$ \001(\002\022@\n\010delegate\030\006 \001"
  "(\0132..mediapipe.InferenceCalculatorOption"
  "s.Delegate2N\n\003ext\022\034.mediapipe.Calculator"
  "Options\030\356\363\274\262\001 \001(\0132\037.mediapipe.FaceDetect"
  "ionOptionsBE\n*com.google.mediapipe.modul"
  "es.facedetectionB\027FaceDetectionFrontProt"
  "o"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_deps[3] = {
  &::descriptor_table_mediapipe_2fcalculators_2ftensor_2finference_5fcalculator_2eproto,
  &::descriptor_table_mediapipe_2fframework_2fcalculator_5foptions_2eproto,
  &::descriptor_table_mediapipe_2fgpu_2fgpu_5forigin_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto = {
  false, false, 761, descriptor_table_protodef_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto, "mediapipe/modules/face_detection/face_detection.proto", 
  &descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_once, descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_deps, 3, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto::offsets,
  file_level_metadata_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto, file_level_enum_descriptors_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto, file_level_service_descriptors_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_getter() {
  return &descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto(&descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto);
namespace mediapipe {

// ===================================================================

class FaceDetectionOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<FaceDetectionOptions>()._has_bits_);
  static void set_has_model_path(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_gpu_origin(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_tensor_width(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_tensor_height(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_num_layers(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_interpolated_scale_aspect_ratio(HasBits* has_bits) {
    (*has_bits)[0] |= 4096u;
  }
  static void set_has_num_boxes(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_x_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_y_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_w_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 512u;
  }
  static void set_has_h_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 1024u;
  }
  static void set_has_min_score_thresh(HasBits* has_bits) {
    (*has_bits)[0] |= 2048u;
  }
  static const ::mediapipe::InferenceCalculatorOptions_Delegate& delegate(const FaceDetectionOptions* msg);
  static void set_has_delegate(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::mediapipe::InferenceCalculatorOptions_Delegate&
FaceDetectionOptions::_Internal::delegate(const FaceDetectionOptions* msg) {
  return *msg->delegate_;
}
void FaceDetectionOptions::clear_delegate() {
  if (delegate_ != nullptr) delegate_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
FaceDetectionOptions::FaceDetectionOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  strides_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.FaceDetectionOptions)
}
FaceDetectionOptions::FaceDetectionOptions(const FaceDetectionOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      strides_(from.strides_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  model_path_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    model_path_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (from._internal_has_model_path()) {
    model_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_model_path(), 
      GetArenaForAllocation());
  }
  if (from._internal_has_delegate()) {
    delegate_ = new ::mediapipe::InferenceCalculatorOptions_Delegate(*from.delegate_);
  } else {
    delegate_ = nullptr;
  }
  ::memcpy(&gpu_origin_, &from.gpu_origin_,
    static_cast<size_t>(reinterpret_cast<char*>(&interpolated_scale_aspect_ratio_) -
    reinterpret_cast<char*>(&gpu_origin_)) + sizeof(interpolated_scale_aspect_ratio_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.FaceDetectionOptions)
}

inline void FaceDetectionOptions::SharedCtor() {
model_path_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  model_path_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&delegate_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&min_score_thresh_) -
    reinterpret_cast<char*>(&delegate_)) + sizeof(min_score_thresh_));
interpolated_scale_aspect_ratio_ = 1;
}

FaceDetectionOptions::~FaceDetectionOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.FaceDetectionOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void FaceDetectionOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  model_path_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (this != internal_default_instance()) delete delegate_;
}

void FaceDetectionOptions::ArenaDtor(void* object) {
  FaceDetectionOptions* _this = reinterpret_cast< FaceDetectionOptions* >(object);
  (void)_this;
}
void FaceDetectionOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void FaceDetectionOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void FaceDetectionOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.FaceDetectionOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  strides_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      model_path_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(delegate_ != nullptr);
      delegate_->Clear();
    }
  }
  if (cached_has_bits & 0x000000fcu) {
    ::memset(&gpu_origin_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&x_scale_) -
        reinterpret_cast<char*>(&gpu_origin_)) + sizeof(x_scale_));
  }
  if (cached_has_bits & 0x00001f00u) {
    ::memset(&y_scale_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&min_score_thresh_) -
        reinterpret_cast<char*>(&y_scale_)) + sizeof(min_score_thresh_));
    interpolated_scale_aspect_ratio_ = 1;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* FaceDetectionOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional string model_path = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_model_path();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.FaceDetectionOptions.model_path");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.InferenceCalculatorOptions.Delegate delegate = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 50)) {
          ptr = ctx->ParseMessage(_internal_mutable_delegate(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.GpuOrigin.Mode gpu_origin = 11;
      case 11:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 88)) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::GpuOrigin_Mode_IsValid(val))) {
            _internal_set_gpu_origin(static_cast<::mediapipe::GpuOrigin_Mode>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(11, val, mutable_unknown_fields());
          }
        } else
          goto handle_unusual;
        continue;
      // optional int32 tensor_width = 21;
      case 21:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 168)) {
          _Internal::set_has_tensor_width(&has_bits);
          tensor_width_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 tensor_height = 22;
      case 22:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 176)) {
          _Internal::set_has_tensor_height(&has_bits);
          tensor_height_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 num_layers = 23;
      case 23:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 184)) {
          _Internal::set_has_num_layers(&has_bits);
          num_layers_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated int32 strides = 24;
      case 24:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 192)) {
          ptr -= 2;
          do {
            ptr += 2;
            _internal_add_strides(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<192>(ptr));
        } else if (static_cast<uint8_t>(tag) == 194) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_strides(), ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional float interpolated_scale_aspect_ratio = 25 [default = 1];
      case 25:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 205)) {
          _Internal::set_has_interpolated_scale_aspect_ratio(&has_bits);
          interpolated_scale_aspect_ratio_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional int32 num_boxes = 31;
      case 31:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 248)) {
          _Internal::set_has_num_boxes(&has_bits);
          num_boxes_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional float x_scale = 32 [default = 0];
      case 32:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 5)) {
          _Internal::set_has_x_scale(&has_bits);
          x_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float y_scale = 33 [default = 0];
      case 33:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 13)) {
          _Internal::set_has_y_scale(&has_bits);
          y_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float w_scale = 34 [default = 0];
      case 34:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 21)) {
          _Internal::set_has_w_scale(&has_bits);
          w_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float h_scale = 35 [default = 0];
      case 35:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 29)) {
          _Internal::set_has_h_scale(&has_bits);
          h_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float min_score_thresh = 36;
      case 36:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 37)) {
          _Internal::set_has_min_score_thresh(&has_bits);
          min_score_thresh_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* FaceDetectionOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.FaceDetectionOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string model_path = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_model_path().data(), static_cast<int>(this->_internal_model_path().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.FaceDetectionOptions.model_path");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_model_path(), target);
  }

  // optional .mediapipe.InferenceCalculatorOptions.Delegate delegate = 6;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        6, _Internal::delegate(this), target, stream);
  }

  // optional .mediapipe.GpuOrigin.Mode gpu_origin = 11;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      11, this->_internal_gpu_origin(), target);
  }

  // optional int32 tensor_width = 21;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(21, this->_internal_tensor_width(), target);
  }

  // optional int32 tensor_height = 22;
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(22, this->_internal_tensor_height(), target);
  }

  // optional int32 num_layers = 23;
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(23, this->_internal_num_layers(), target);
  }

  // repeated int32 strides = 24;
  for (int i = 0, n = this->_internal_strides_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(24, this->_internal_strides(i), target);
  }

  // optional float interpolated_scale_aspect_ratio = 25 [default = 1];
  if (cached_has_bits & 0x00001000u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(25, this->_internal_interpolated_scale_aspect_ratio(), target);
  }

  // optional int32 num_boxes = 31;
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(31, this->_internal_num_boxes(), target);
  }

  // optional float x_scale = 32 [default = 0];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(32, this->_internal_x_scale(), target);
  }

  // optional float y_scale = 33 [default = 0];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(33, this->_internal_y_scale(), target);
  }

  // optional float w_scale = 34 [default = 0];
  if (cached_has_bits & 0x00000200u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(34, this->_internal_w_scale(), target);
  }

  // optional float h_scale = 35 [default = 0];
  if (cached_has_bits & 0x00000400u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(35, this->_internal_h_scale(), target);
  }

  // optional float min_score_thresh = 36;
  if (cached_has_bits & 0x00000800u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(36, this->_internal_min_score_thresh(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.FaceDetectionOptions)
  return target;
}

size_t FaceDetectionOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.FaceDetectionOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 strides = 24;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->strides_);
    total_size += 2 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_strides_size());
    total_size += data_size;
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional string model_path = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_model_path());
    }

    // optional .mediapipe.InferenceCalculatorOptions.Delegate delegate = 6;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *delegate_);
    }

    // optional .mediapipe.GpuOrigin.Mode gpu_origin = 11;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_gpu_origin());
    }

    // optional int32 tensor_width = 21;
    if (cached_has_bits & 0x00000008u) {
      total_size += 2 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_tensor_width());
    }

    // optional int32 tensor_height = 22;
    if (cached_has_bits & 0x00000010u) {
      total_size += 2 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_tensor_height());
    }

    // optional int32 num_layers = 23;
    if (cached_has_bits & 0x00000020u) {
      total_size += 2 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_num_layers());
    }

    // optional int32 num_boxes = 31;
    if (cached_has_bits & 0x00000040u) {
      total_size += 2 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_num_boxes());
    }

    // optional float x_scale = 32 [default = 0];
    if (cached_has_bits & 0x00000080u) {
      total_size += 2 + 4;
    }

  }
  if (cached_has_bits & 0x00001f00u) {
    // optional float y_scale = 33 [default = 0];
    if (cached_has_bits & 0x00000100u) {
      total_size += 2 + 4;
    }

    // optional float w_scale = 34 [default = 0];
    if (cached_has_bits & 0x00000200u) {
      total_size += 2 + 4;
    }

    // optional float h_scale = 35 [default = 0];
    if (cached_has_bits & 0x00000400u) {
      total_size += 2 + 4;
    }

    // optional float min_score_thresh = 36;
    if (cached_has_bits & 0x00000800u) {
      total_size += 2 + 4;
    }

    // optional float interpolated_scale_aspect_ratio = 25 [default = 1];
    if (cached_has_bits & 0x00001000u) {
      total_size += 2 + 4;
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData FaceDetectionOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    FaceDetectionOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*FaceDetectionOptions::GetClassData() const { return &_class_data_; }

void FaceDetectionOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<FaceDetectionOptions *>(to)->MergeFrom(
      static_cast<const FaceDetectionOptions &>(from));
}


void FaceDetectionOptions::MergeFrom(const FaceDetectionOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.FaceDetectionOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  strides_.MergeFrom(from.strides_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_model_path(from._internal_model_path());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_delegate()->::mediapipe::InferenceCalculatorOptions_Delegate::MergeFrom(from._internal_delegate());
    }
    if (cached_has_bits & 0x00000004u) {
      gpu_origin_ = from.gpu_origin_;
    }
    if (cached_has_bits & 0x00000008u) {
      tensor_width_ = from.tensor_width_;
    }
    if (cached_has_bits & 0x00000010u) {
      tensor_height_ = from.tensor_height_;
    }
    if (cached_has_bits & 0x00000020u) {
      num_layers_ = from.num_layers_;
    }
    if (cached_has_bits & 0x00000040u) {
      num_boxes_ = from.num_boxes_;
    }
    if (cached_has_bits & 0x00000080u) {
      x_scale_ = from.x_scale_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00001f00u) {
    if (cached_has_bits & 0x00000100u) {
      y_scale_ = from.y_scale_;
    }
    if (cached_has_bits & 0x00000200u) {
      w_scale_ = from.w_scale_;
    }
    if (cached_has_bits & 0x00000400u) {
      h_scale_ = from.h_scale_;
    }
    if (cached_has_bits & 0x00000800u) {
      min_score_thresh_ = from.min_score_thresh_;
    }
    if (cached_has_bits & 0x00001000u) {
      interpolated_scale_aspect_ratio_ = from.interpolated_scale_aspect_ratio_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void FaceDetectionOptions::CopyFrom(const FaceDetectionOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.FaceDetectionOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool FaceDetectionOptions::IsInitialized() const {
  return true;
}

void FaceDetectionOptions::InternalSwap(FaceDetectionOptions* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  strides_.InternalSwap(&other->strides_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &model_path_, lhs_arena,
      &other->model_path_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(FaceDetectionOptions, min_score_thresh_)
      + sizeof(FaceDetectionOptions::min_score_thresh_)
      - PROTOBUF_FIELD_OFFSET(FaceDetectionOptions, delegate_)>(
          reinterpret_cast<char*>(&delegate_),
          reinterpret_cast<char*>(&other->delegate_));
  swap(interpolated_scale_aspect_ratio_, other->interpolated_scale_aspect_ratio_);
}

::PROTOBUF_NAMESPACE_ID::Metadata FaceDetectionOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_getter, &descriptor_table_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto_once,
      file_level_metadata_mediapipe_2fmodules_2fface_5fdetection_2fface_5fdetection_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int FaceDetectionOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FaceDetectionOptions >, 11, false >
  FaceDetectionOptions::ext(kExtFieldNumber, ::mediapipe::FaceDetectionOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::FaceDetectionOptions* Arena::CreateMaybeMessage< ::mediapipe::FaceDetectionOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::FaceDetectionOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>