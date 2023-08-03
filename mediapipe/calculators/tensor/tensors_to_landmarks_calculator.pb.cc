// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"

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
constexpr TensorsToLandmarksCalculatorOptions::TensorsToLandmarksCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : num_landmarks_(0)
  , input_image_width_(0)
  , input_image_height_(0)
  , flip_vertically_(false)
  , flip_horizontally_(false)
  , visibility_activation_(0)

  , presence_activation_(0)

  , normalize_z_(1){}
struct TensorsToLandmarksCalculatorOptionsDefaultTypeInternal {
  constexpr TensorsToLandmarksCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~TensorsToLandmarksCalculatorOptionsDefaultTypeInternal() {}
  union {
    TensorsToLandmarksCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT TensorsToLandmarksCalculatorOptionsDefaultTypeInternal _TensorsToLandmarksCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, num_landmarks_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, input_image_width_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, input_image_height_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, flip_vertically_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, flip_horizontally_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, normalize_z_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, visibility_activation_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToLandmarksCalculatorOptions, presence_activation_),
  0,
  1,
  2,
  3,
  4,
  7,
  5,
  6,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 14, -1, sizeof(::mediapipe::TensorsToLandmarksCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_TensorsToLandmarksCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nBmediapipe/calculators/tensor/tensors_t"
  "o_landmarks_calculator.proto\022\tmediapipe\032"
  "$mediapipe/framework/calculator.proto\"\217\004"
  "\n#TensorsToLandmarksCalculatorOptions\022\025\n"
  "\rnum_landmarks\030\001 \001(\005\022\031\n\021input_image_widt"
  "h\030\002 \001(\005\022\032\n\022input_image_height\030\003 \001(\005\022\036\n\017f"
  "lip_vertically\030\004 \001(\010:\005false\022 \n\021flip_hori"
  "zontally\030\006 \001(\010:\005false\022\026\n\013normalize_z\030\005 \001"
  "(\002:\0011\022^\n\025visibility_activation\030\007 \001(\01629.m"
  "ediapipe.TensorsToLandmarksCalculatorOpt"
  "ions.Activation:\004NONE\022\\\n\023presence_activa"
  "tion\030\010 \001(\01629.mediapipe.TensorsToLandmark"
  "sCalculatorOptions.Activation:\004NONE\"#\n\nA"
  "ctivation\022\010\n\004NONE\020\000\022\013\n\007SIGMOID\020\0012]\n\003ext\022"
  "\034.mediapipe.CalculatorOptions\030\260\215\214\240\001 \001(\0132"
  "..mediapipe.TensorsToLandmarksCalculator"
  "Options"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto = {
  false, false, 647, descriptor_table_protodef_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto, "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* TensorsToLandmarksCalculatorOptions_Activation_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto);
  return file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto[0];
}
bool TensorsToLandmarksCalculatorOptions_Activation_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))
constexpr TensorsToLandmarksCalculatorOptions_Activation TensorsToLandmarksCalculatorOptions::NONE;
constexpr TensorsToLandmarksCalculatorOptions_Activation TensorsToLandmarksCalculatorOptions::SIGMOID;
constexpr TensorsToLandmarksCalculatorOptions_Activation TensorsToLandmarksCalculatorOptions::Activation_MIN;
constexpr TensorsToLandmarksCalculatorOptions_Activation TensorsToLandmarksCalculatorOptions::Activation_MAX;
constexpr int TensorsToLandmarksCalculatorOptions::Activation_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))

// ===================================================================

class TensorsToLandmarksCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<TensorsToLandmarksCalculatorOptions>()._has_bits_);
  static void set_has_num_landmarks(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_input_image_width(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_input_image_height(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_flip_vertically(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_flip_horizontally(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_normalize_z(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_visibility_activation(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_presence_activation(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
};

TensorsToLandmarksCalculatorOptions::TensorsToLandmarksCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.TensorsToLandmarksCalculatorOptions)
}
TensorsToLandmarksCalculatorOptions::TensorsToLandmarksCalculatorOptions(const TensorsToLandmarksCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&num_landmarks_, &from.num_landmarks_,
    static_cast<size_t>(reinterpret_cast<char*>(&normalize_z_) -
    reinterpret_cast<char*>(&num_landmarks_)) + sizeof(normalize_z_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.TensorsToLandmarksCalculatorOptions)
}

inline void TensorsToLandmarksCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&num_landmarks_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&presence_activation_) -
    reinterpret_cast<char*>(&num_landmarks_)) + sizeof(presence_activation_));
normalize_z_ = 1;
}

TensorsToLandmarksCalculatorOptions::~TensorsToLandmarksCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.TensorsToLandmarksCalculatorOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void TensorsToLandmarksCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void TensorsToLandmarksCalculatorOptions::ArenaDtor(void* object) {
  TensorsToLandmarksCalculatorOptions* _this = reinterpret_cast< TensorsToLandmarksCalculatorOptions* >(object);
  (void)_this;
}
void TensorsToLandmarksCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TensorsToLandmarksCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TensorsToLandmarksCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.TensorsToLandmarksCalculatorOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    ::memset(&num_landmarks_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&presence_activation_) -
        reinterpret_cast<char*>(&num_landmarks_)) + sizeof(presence_activation_));
    normalize_z_ = 1;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TensorsToLandmarksCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 num_landmarks = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _Internal::set_has_num_landmarks(&has_bits);
          num_landmarks_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 input_image_width = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 16)) {
          _Internal::set_has_input_image_width(&has_bits);
          input_image_width_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 input_image_height = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 24)) {
          _Internal::set_has_input_image_height(&has_bits);
          input_image_height_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional bool flip_vertically = 4 [default = false];
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 32)) {
          _Internal::set_has_flip_vertically(&has_bits);
          flip_vertically_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional float normalize_z = 5 [default = 1];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 45)) {
          _Internal::set_has_normalize_z(&has_bits);
          normalize_z_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional bool flip_horizontally = 6 [default = false];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 48)) {
          _Internal::set_has_flip_horizontally(&has_bits);
          flip_horizontally_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.TensorsToLandmarksCalculatorOptions.Activation visibility_activation = 7 [default = NONE];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 56)) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::TensorsToLandmarksCalculatorOptions_Activation_IsValid(val))) {
            _internal_set_visibility_activation(static_cast<::mediapipe::TensorsToLandmarksCalculatorOptions_Activation>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(7, val, mutable_unknown_fields());
          }
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.TensorsToLandmarksCalculatorOptions.Activation presence_activation = 8 [default = NONE];
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 64)) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::TensorsToLandmarksCalculatorOptions_Activation_IsValid(val))) {
            _internal_set_presence_activation(static_cast<::mediapipe::TensorsToLandmarksCalculatorOptions_Activation>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(8, val, mutable_unknown_fields());
          }
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

uint8_t* TensorsToLandmarksCalculatorOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.TensorsToLandmarksCalculatorOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 num_landmarks = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_num_landmarks(), target);
  }

  // optional int32 input_image_width = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_input_image_width(), target);
  }

  // optional int32 input_image_height = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_input_image_height(), target);
  }

  // optional bool flip_vertically = 4 [default = false];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(4, this->_internal_flip_vertically(), target);
  }

  // optional float normalize_z = 5 [default = 1];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(5, this->_internal_normalize_z(), target);
  }

  // optional bool flip_horizontally = 6 [default = false];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(6, this->_internal_flip_horizontally(), target);
  }

  // optional .mediapipe.TensorsToLandmarksCalculatorOptions.Activation visibility_activation = 7 [default = NONE];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      7, this->_internal_visibility_activation(), target);
  }

  // optional .mediapipe.TensorsToLandmarksCalculatorOptions.Activation presence_activation = 8 [default = NONE];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      8, this->_internal_presence_activation(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.TensorsToLandmarksCalculatorOptions)
  return target;
}

size_t TensorsToLandmarksCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.TensorsToLandmarksCalculatorOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional int32 num_landmarks = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_num_landmarks());
    }

    // optional int32 input_image_width = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_input_image_width());
    }

    // optional int32 input_image_height = 3;
    if (cached_has_bits & 0x00000004u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_input_image_height());
    }

    // optional bool flip_vertically = 4 [default = false];
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 1;
    }

    // optional bool flip_horizontally = 6 [default = false];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 1;
    }

    // optional .mediapipe.TensorsToLandmarksCalculatorOptions.Activation visibility_activation = 7 [default = NONE];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_visibility_activation());
    }

    // optional .mediapipe.TensorsToLandmarksCalculatorOptions.Activation presence_activation = 8 [default = NONE];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_presence_activation());
    }

    // optional float normalize_z = 5 [default = 1];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 + 4;
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData TensorsToLandmarksCalculatorOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    TensorsToLandmarksCalculatorOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*TensorsToLandmarksCalculatorOptions::GetClassData() const { return &_class_data_; }

void TensorsToLandmarksCalculatorOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<TensorsToLandmarksCalculatorOptions *>(to)->MergeFrom(
      static_cast<const TensorsToLandmarksCalculatorOptions &>(from));
}


void TensorsToLandmarksCalculatorOptions::MergeFrom(const TensorsToLandmarksCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.TensorsToLandmarksCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      num_landmarks_ = from.num_landmarks_;
    }
    if (cached_has_bits & 0x00000002u) {
      input_image_width_ = from.input_image_width_;
    }
    if (cached_has_bits & 0x00000004u) {
      input_image_height_ = from.input_image_height_;
    }
    if (cached_has_bits & 0x00000008u) {
      flip_vertically_ = from.flip_vertically_;
    }
    if (cached_has_bits & 0x00000010u) {
      flip_horizontally_ = from.flip_horizontally_;
    }
    if (cached_has_bits & 0x00000020u) {
      visibility_activation_ = from.visibility_activation_;
    }
    if (cached_has_bits & 0x00000040u) {
      presence_activation_ = from.presence_activation_;
    }
    if (cached_has_bits & 0x00000080u) {
      normalize_z_ = from.normalize_z_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void TensorsToLandmarksCalculatorOptions::CopyFrom(const TensorsToLandmarksCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.TensorsToLandmarksCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorsToLandmarksCalculatorOptions::IsInitialized() const {
  return true;
}

void TensorsToLandmarksCalculatorOptions::InternalSwap(TensorsToLandmarksCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(TensorsToLandmarksCalculatorOptions, presence_activation_)
      + sizeof(TensorsToLandmarksCalculatorOptions::presence_activation_)
      - PROTOBUF_FIELD_OFFSET(TensorsToLandmarksCalculatorOptions, num_landmarks_)>(
          reinterpret_cast<char*>(&num_landmarks_),
          reinterpret_cast<char*>(&other->num_landmarks_));
  swap(normalize_z_, other->normalize_z_);
}

::PROTOBUF_NAMESPACE_ID::Metadata TensorsToLandmarksCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5flandmarks_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int TensorsToLandmarksCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TensorsToLandmarksCalculatorOptions >, 11, false >
  TensorsToLandmarksCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::TensorsToLandmarksCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::TensorsToLandmarksCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::TensorsToLandmarksCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::TensorsToLandmarksCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
