// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tflite/ssd_anchors_calculator.proto

#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"

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
constexpr SsdAnchorsCalculatorOptions::SsdAnchorsCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : feature_map_width_()
  , feature_map_height_()
  , strides_()
  , aspect_ratios_()
  , input_size_width_(0)
  , input_size_height_(0)
  , min_scale_(0)
  , max_scale_(0)
  , num_layers_(0)
  , reduce_boxes_in_lowest_layer_(false)
  , fixed_anchor_size_(false)
  , interpolated_scale_aspect_ratio_(1)
  , anchor_offset_x_(0.5f)
  , anchor_offset_y_(0.5f){}
struct SsdAnchorsCalculatorOptionsDefaultTypeInternal {
  constexpr SsdAnchorsCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SsdAnchorsCalculatorOptionsDefaultTypeInternal() {}
  union {
    SsdAnchorsCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SsdAnchorsCalculatorOptionsDefaultTypeInternal _SsdAnchorsCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, input_size_width_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, input_size_height_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, min_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, max_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, anchor_offset_x_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, anchor_offset_y_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, num_layers_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, feature_map_width_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, feature_map_height_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, strides_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, aspect_ratios_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, reduce_boxes_in_lowest_layer_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, interpolated_scale_aspect_ratio_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SsdAnchorsCalculatorOptions, fixed_anchor_size_),
  0,
  1,
  2,
  3,
  8,
  9,
  4,
  ~0u,
  ~0u,
  ~0u,
  ~0u,
  5,
  7,
  6,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 20, -1, sizeof(::mediapipe::SsdAnchorsCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_SsdAnchorsCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n9mediapipe/calculators/tflite/ssd_ancho"
  "rs_calculator.proto\022\tmediapipe\032$mediapip"
  "e/framework/calculator.proto\"\370\003\n\033SsdAnch"
  "orsCalculatorOptions\022\030\n\020input_size_width"
  "\030\001 \001(\005\022\031\n\021input_size_height\030\002 \001(\005\022\021\n\tmin"
  "_scale\030\003 \001(\002\022\021\n\tmax_scale\030\004 \001(\002\022\034\n\017ancho"
  "r_offset_x\030\005 \001(\002:\0030.5\022\034\n\017anchor_offset_y"
  "\030\006 \001(\002:\0030.5\022\022\n\nnum_layers\030\007 \001(\005\022\031\n\021featu"
  "re_map_width\030\010 \003(\005\022\032\n\022feature_map_height"
  "\030\t \003(\005\022\017\n\007strides\030\n \003(\005\022\025\n\raspect_ratios"
  "\030\013 \003(\002\022+\n\034reduce_boxes_in_lowest_layer\030\014"
  " \001(\010:\005false\022*\n\037interpolated_scale_aspect"
  "_ratio\030\r \001(\002:\0011\022 \n\021fixed_anchor_size\030\016 \001"
  "(\010:\005false2T\n\003ext\022\034.mediapipe.CalculatorO"
  "ptions\030\377\270\363u \001(\0132&.mediapipe.SsdAnchorsCa"
  "lculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto = {
  false, false, 615, descriptor_table_protodef_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto, "mediapipe/calculators/tflite/ssd_anchors_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class SsdAnchorsCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<SsdAnchorsCalculatorOptions>()._has_bits_);
  static void set_has_input_size_width(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_input_size_height(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_min_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_max_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_anchor_offset_x(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_anchor_offset_y(HasBits* has_bits) {
    (*has_bits)[0] |= 512u;
  }
  static void set_has_num_layers(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_reduce_boxes_in_lowest_layer(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_interpolated_scale_aspect_ratio(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_fixed_anchor_size(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
};

SsdAnchorsCalculatorOptions::SsdAnchorsCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  feature_map_width_(arena),
  feature_map_height_(arena),
  strides_(arena),
  aspect_ratios_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.SsdAnchorsCalculatorOptions)
}
SsdAnchorsCalculatorOptions::SsdAnchorsCalculatorOptions(const SsdAnchorsCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      feature_map_width_(from.feature_map_width_),
      feature_map_height_(from.feature_map_height_),
      strides_(from.strides_),
      aspect_ratios_(from.aspect_ratios_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&input_size_width_, &from.input_size_width_,
    static_cast<size_t>(reinterpret_cast<char*>(&anchor_offset_y_) -
    reinterpret_cast<char*>(&input_size_width_)) + sizeof(anchor_offset_y_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.SsdAnchorsCalculatorOptions)
}

inline void SsdAnchorsCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&input_size_width_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&fixed_anchor_size_) -
    reinterpret_cast<char*>(&input_size_width_)) + sizeof(fixed_anchor_size_));
interpolated_scale_aspect_ratio_ = 1;
anchor_offset_x_ = 0.5f;
anchor_offset_y_ = 0.5f;
}

SsdAnchorsCalculatorOptions::~SsdAnchorsCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.SsdAnchorsCalculatorOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void SsdAnchorsCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void SsdAnchorsCalculatorOptions::ArenaDtor(void* object) {
  SsdAnchorsCalculatorOptions* _this = reinterpret_cast< SsdAnchorsCalculatorOptions* >(object);
  (void)_this;
}
void SsdAnchorsCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SsdAnchorsCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SsdAnchorsCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.SsdAnchorsCalculatorOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  feature_map_width_.Clear();
  feature_map_height_.Clear();
  strides_.Clear();
  aspect_ratios_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    ::memset(&input_size_width_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&fixed_anchor_size_) -
        reinterpret_cast<char*>(&input_size_width_)) + sizeof(fixed_anchor_size_));
    interpolated_scale_aspect_ratio_ = 1;
  }
  if (cached_has_bits & 0x00000300u) {
    anchor_offset_x_ = 0.5f;
    anchor_offset_y_ = 0.5f;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SsdAnchorsCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 input_size_width = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _Internal::set_has_input_size_width(&has_bits);
          input_size_width_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 input_size_height = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 16)) {
          _Internal::set_has_input_size_height(&has_bits);
          input_size_height_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional float min_scale = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 29)) {
          _Internal::set_has_min_scale(&has_bits);
          min_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float max_scale = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 37)) {
          _Internal::set_has_max_scale(&has_bits);
          max_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float anchor_offset_x = 5 [default = 0.5];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 45)) {
          _Internal::set_has_anchor_offset_x(&has_bits);
          anchor_offset_x_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional float anchor_offset_y = 6 [default = 0.5];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 53)) {
          _Internal::set_has_anchor_offset_y(&has_bits);
          anchor_offset_y_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional int32 num_layers = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 56)) {
          _Internal::set_has_num_layers(&has_bits);
          num_layers_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated int32 feature_map_width = 8;
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 64)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_feature_map_width(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<64>(ptr));
        } else if (static_cast<uint8_t>(tag) == 66) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_feature_map_width(), ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated int32 feature_map_height = 9;
      case 9:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 72)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_feature_map_height(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<72>(ptr));
        } else if (static_cast<uint8_t>(tag) == 74) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_feature_map_height(), ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated int32 strides = 10;
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 80)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_strides(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<80>(ptr));
        } else if (static_cast<uint8_t>(tag) == 82) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_strides(), ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated float aspect_ratios = 11;
      case 11:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 93)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_aspect_ratios(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr));
            ptr += sizeof(float);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<93>(ptr));
        } else if (static_cast<uint8_t>(tag) == 90) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedFloatParser(_internal_mutable_aspect_ratios(), ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional bool reduce_boxes_in_lowest_layer = 12 [default = false];
      case 12:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 96)) {
          _Internal::set_has_reduce_boxes_in_lowest_layer(&has_bits);
          reduce_boxes_in_lowest_layer_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional float interpolated_scale_aspect_ratio = 13 [default = 1];
      case 13:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 109)) {
          _Internal::set_has_interpolated_scale_aspect_ratio(&has_bits);
          interpolated_scale_aspect_ratio_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // optional bool fixed_anchor_size = 14 [default = false];
      case 14:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 112)) {
          _Internal::set_has_fixed_anchor_size(&has_bits);
          fixed_anchor_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
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

uint8_t* SsdAnchorsCalculatorOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.SsdAnchorsCalculatorOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 input_size_width = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_input_size_width(), target);
  }

  // optional int32 input_size_height = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_input_size_height(), target);
  }

  // optional float min_scale = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(3, this->_internal_min_scale(), target);
  }

  // optional float max_scale = 4;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(4, this->_internal_max_scale(), target);
  }

  // optional float anchor_offset_x = 5 [default = 0.5];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(5, this->_internal_anchor_offset_x(), target);
  }

  // optional float anchor_offset_y = 6 [default = 0.5];
  if (cached_has_bits & 0x00000200u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(6, this->_internal_anchor_offset_y(), target);
  }

  // optional int32 num_layers = 7;
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(7, this->_internal_num_layers(), target);
  }

  // repeated int32 feature_map_width = 8;
  for (int i = 0, n = this->_internal_feature_map_width_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(8, this->_internal_feature_map_width(i), target);
  }

  // repeated int32 feature_map_height = 9;
  for (int i = 0, n = this->_internal_feature_map_height_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(9, this->_internal_feature_map_height(i), target);
  }

  // repeated int32 strides = 10;
  for (int i = 0, n = this->_internal_strides_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(10, this->_internal_strides(i), target);
  }

  // repeated float aspect_ratios = 11;
  for (int i = 0, n = this->_internal_aspect_ratios_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(11, this->_internal_aspect_ratios(i), target);
  }

  // optional bool reduce_boxes_in_lowest_layer = 12 [default = false];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(12, this->_internal_reduce_boxes_in_lowest_layer(), target);
  }

  // optional float interpolated_scale_aspect_ratio = 13 [default = 1];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(13, this->_internal_interpolated_scale_aspect_ratio(), target);
  }

  // optional bool fixed_anchor_size = 14 [default = false];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(14, this->_internal_fixed_anchor_size(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.SsdAnchorsCalculatorOptions)
  return target;
}

size_t SsdAnchorsCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.SsdAnchorsCalculatorOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 feature_map_width = 8;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->feature_map_width_);
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_feature_map_width_size());
    total_size += data_size;
  }

  // repeated int32 feature_map_height = 9;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->feature_map_height_);
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_feature_map_height_size());
    total_size += data_size;
  }

  // repeated int32 strides = 10;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->strides_);
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_strides_size());
    total_size += data_size;
  }

  // repeated float aspect_ratios = 11;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_aspect_ratios_size());
    size_t data_size = 4UL * count;
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_aspect_ratios_size());
    total_size += data_size;
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional int32 input_size_width = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_input_size_width());
    }

    // optional int32 input_size_height = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_input_size_height());
    }

    // optional float min_scale = 3;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 + 4;
    }

    // optional float max_scale = 4;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 4;
    }

    // optional int32 num_layers = 7;
    if (cached_has_bits & 0x00000010u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_num_layers());
    }

    // optional bool reduce_boxes_in_lowest_layer = 12 [default = false];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 + 1;
    }

    // optional bool fixed_anchor_size = 14 [default = false];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 + 1;
    }

    // optional float interpolated_scale_aspect_ratio = 13 [default = 1];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 + 4;
    }

  }
  if (cached_has_bits & 0x00000300u) {
    // optional float anchor_offset_x = 5 [default = 0.5];
    if (cached_has_bits & 0x00000100u) {
      total_size += 1 + 4;
    }

    // optional float anchor_offset_y = 6 [default = 0.5];
    if (cached_has_bits & 0x00000200u) {
      total_size += 1 + 4;
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SsdAnchorsCalculatorOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    SsdAnchorsCalculatorOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SsdAnchorsCalculatorOptions::GetClassData() const { return &_class_data_; }

void SsdAnchorsCalculatorOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<SsdAnchorsCalculatorOptions *>(to)->MergeFrom(
      static_cast<const SsdAnchorsCalculatorOptions &>(from));
}


void SsdAnchorsCalculatorOptions::MergeFrom(const SsdAnchorsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.SsdAnchorsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  feature_map_width_.MergeFrom(from.feature_map_width_);
  feature_map_height_.MergeFrom(from.feature_map_height_);
  strides_.MergeFrom(from.strides_);
  aspect_ratios_.MergeFrom(from.aspect_ratios_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      input_size_width_ = from.input_size_width_;
    }
    if (cached_has_bits & 0x00000002u) {
      input_size_height_ = from.input_size_height_;
    }
    if (cached_has_bits & 0x00000004u) {
      min_scale_ = from.min_scale_;
    }
    if (cached_has_bits & 0x00000008u) {
      max_scale_ = from.max_scale_;
    }
    if (cached_has_bits & 0x00000010u) {
      num_layers_ = from.num_layers_;
    }
    if (cached_has_bits & 0x00000020u) {
      reduce_boxes_in_lowest_layer_ = from.reduce_boxes_in_lowest_layer_;
    }
    if (cached_has_bits & 0x00000040u) {
      fixed_anchor_size_ = from.fixed_anchor_size_;
    }
    if (cached_has_bits & 0x00000080u) {
      interpolated_scale_aspect_ratio_ = from.interpolated_scale_aspect_ratio_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00000300u) {
    if (cached_has_bits & 0x00000100u) {
      anchor_offset_x_ = from.anchor_offset_x_;
    }
    if (cached_has_bits & 0x00000200u) {
      anchor_offset_y_ = from.anchor_offset_y_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SsdAnchorsCalculatorOptions::CopyFrom(const SsdAnchorsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.SsdAnchorsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SsdAnchorsCalculatorOptions::IsInitialized() const {
  return true;
}

void SsdAnchorsCalculatorOptions::InternalSwap(SsdAnchorsCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  feature_map_width_.InternalSwap(&other->feature_map_width_);
  feature_map_height_.InternalSwap(&other->feature_map_height_);
  strides_.InternalSwap(&other->strides_);
  aspect_ratios_.InternalSwap(&other->aspect_ratios_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(SsdAnchorsCalculatorOptions, fixed_anchor_size_)
      + sizeof(SsdAnchorsCalculatorOptions::fixed_anchor_size_)
      - PROTOBUF_FIELD_OFFSET(SsdAnchorsCalculatorOptions, input_size_width_)>(
          reinterpret_cast<char*>(&input_size_width_),
          reinterpret_cast<char*>(&other->input_size_width_));
  swap(interpolated_scale_aspect_ratio_, other->interpolated_scale_aspect_ratio_);
  swap(anchor_offset_x_, other->anchor_offset_x_);
  swap(anchor_offset_y_, other->anchor_offset_y_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SsdAnchorsCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftflite_2fssd_5fanchors_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int SsdAnchorsCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::SsdAnchorsCalculatorOptions >, 11, false >
  SsdAnchorsCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::SsdAnchorsCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::SsdAnchorsCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::SsdAnchorsCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::SsdAnchorsCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>