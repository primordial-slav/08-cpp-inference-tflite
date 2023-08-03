// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/collection_has_min_size_calculator.proto

#include "mediapipe/calculators/util/collection_has_min_size_calculator.pb.h"

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
constexpr CollectionHasMinSizeCalculatorOptions::CollectionHasMinSizeCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : min_size_(0){}
struct CollectionHasMinSizeCalculatorOptionsDefaultTypeInternal {
  constexpr CollectionHasMinSizeCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~CollectionHasMinSizeCalculatorOptionsDefaultTypeInternal() {}
  union {
    CollectionHasMinSizeCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT CollectionHasMinSizeCalculatorOptionsDefaultTypeInternal _CollectionHasMinSizeCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::CollectionHasMinSizeCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::CollectionHasMinSizeCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::CollectionHasMinSizeCalculatorOptions, min_size_),
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, -1, sizeof(::mediapipe::CollectionHasMinSizeCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_CollectionHasMinSizeCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nCmediapipe/calculators/util/collection_"
  "has_min_size_calculator.proto\022\tmediapipe"
  "\032$mediapipe/framework/calculator.proto\"\234"
  "\001\n%CollectionHasMinSizeCalculatorOptions"
  "\022\023\n\010min_size\030\001 \001(\005:\00102^\n\003ext\022\034.mediapipe"
  ".CalculatorOptions\030\320\261\330{ \001(\01320.mediapipe."
  "CollectionHasMinSizeCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto = {
  false, false, 277, descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto, "mediapipe/calculators/util/collection_has_min_size_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class CollectionHasMinSizeCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<CollectionHasMinSizeCalculatorOptions>()._has_bits_);
  static void set_has_min_size(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

CollectionHasMinSizeCalculatorOptions::CollectionHasMinSizeCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.CollectionHasMinSizeCalculatorOptions)
}
CollectionHasMinSizeCalculatorOptions::CollectionHasMinSizeCalculatorOptions(const CollectionHasMinSizeCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  min_size_ = from.min_size_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.CollectionHasMinSizeCalculatorOptions)
}

inline void CollectionHasMinSizeCalculatorOptions::SharedCtor() {
min_size_ = 0;
}

CollectionHasMinSizeCalculatorOptions::~CollectionHasMinSizeCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.CollectionHasMinSizeCalculatorOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void CollectionHasMinSizeCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void CollectionHasMinSizeCalculatorOptions::ArenaDtor(void* object) {
  CollectionHasMinSizeCalculatorOptions* _this = reinterpret_cast< CollectionHasMinSizeCalculatorOptions* >(object);
  (void)_this;
}
void CollectionHasMinSizeCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void CollectionHasMinSizeCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void CollectionHasMinSizeCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.CollectionHasMinSizeCalculatorOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  min_size_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* CollectionHasMinSizeCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 min_size = 1 [default = 0];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _Internal::set_has_min_size(&has_bits);
          min_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
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

uint8_t* CollectionHasMinSizeCalculatorOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.CollectionHasMinSizeCalculatorOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 min_size = 1 [default = 0];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_min_size(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.CollectionHasMinSizeCalculatorOptions)
  return target;
}

size_t CollectionHasMinSizeCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.CollectionHasMinSizeCalculatorOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // optional int32 min_size = 1 [default = 0];
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_min_size());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData CollectionHasMinSizeCalculatorOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    CollectionHasMinSizeCalculatorOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*CollectionHasMinSizeCalculatorOptions::GetClassData() const { return &_class_data_; }

void CollectionHasMinSizeCalculatorOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<CollectionHasMinSizeCalculatorOptions *>(to)->MergeFrom(
      static_cast<const CollectionHasMinSizeCalculatorOptions &>(from));
}


void CollectionHasMinSizeCalculatorOptions::MergeFrom(const CollectionHasMinSizeCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.CollectionHasMinSizeCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_min_size()) {
    _internal_set_min_size(from._internal_min_size());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void CollectionHasMinSizeCalculatorOptions::CopyFrom(const CollectionHasMinSizeCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.CollectionHasMinSizeCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool CollectionHasMinSizeCalculatorOptions::IsInitialized() const {
  return true;
}

void CollectionHasMinSizeCalculatorOptions::InternalSwap(CollectionHasMinSizeCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(min_size_, other->min_size_);
}

::PROTOBUF_NAMESPACE_ID::Metadata CollectionHasMinSizeCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int CollectionHasMinSizeCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::CollectionHasMinSizeCalculatorOptions >, 11, false >
  CollectionHasMinSizeCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::CollectionHasMinSizeCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::CollectionHasMinSizeCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::CollectionHasMinSizeCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::CollectionHasMinSizeCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
