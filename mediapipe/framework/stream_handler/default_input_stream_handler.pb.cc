// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/stream_handler/default_input_stream_handler.proto

#include "mediapipe/framework/stream_handler/default_input_stream_handler.pb.h"

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
constexpr DefaultInputStreamHandlerOptions::DefaultInputStreamHandlerOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : batch_size_(1){}
struct DefaultInputStreamHandlerOptionsDefaultTypeInternal {
  constexpr DefaultInputStreamHandlerOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DefaultInputStreamHandlerOptionsDefaultTypeInternal() {}
  union {
    DefaultInputStreamHandlerOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DefaultInputStreamHandlerOptionsDefaultTypeInternal _DefaultInputStreamHandlerOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::DefaultInputStreamHandlerOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DefaultInputStreamHandlerOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::DefaultInputStreamHandlerOptions, batch_size_),
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, -1, sizeof(::mediapipe::DefaultInputStreamHandlerOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_DefaultInputStreamHandlerOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nEmediapipe/framework/stream_handler/def"
  "ault_input_stream_handler.proto\022\tmediapi"
  "pe\032+mediapipe/framework/mediapipe_option"
  "s.proto\"\223\001\n DefaultInputStreamHandlerOpt"
  "ions\022\025\n\nbatch_size\030\001 \001(\005:\00112X\n\003ext\022\033.med"
  "iapipe.MediaPipeOptions\030\365\355\254N \001(\0132+.media"
  "pipe.DefaultInputStreamHandlerOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fmediapipe_5foptions_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto = {
  false, false, 277, descriptor_table_protodef_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto, "mediapipe/framework/stream_handler/default_input_stream_handler.proto", 
  &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_once, descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto, file_level_service_descriptors_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto(&descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto);
namespace mediapipe {

// ===================================================================

class DefaultInputStreamHandlerOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<DefaultInputStreamHandlerOptions>()._has_bits_);
  static void set_has_batch_size(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

DefaultInputStreamHandlerOptions::DefaultInputStreamHandlerOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.DefaultInputStreamHandlerOptions)
}
DefaultInputStreamHandlerOptions::DefaultInputStreamHandlerOptions(const DefaultInputStreamHandlerOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  batch_size_ = from.batch_size_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.DefaultInputStreamHandlerOptions)
}

inline void DefaultInputStreamHandlerOptions::SharedCtor() {
batch_size_ = 1;
}

DefaultInputStreamHandlerOptions::~DefaultInputStreamHandlerOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.DefaultInputStreamHandlerOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void DefaultInputStreamHandlerOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void DefaultInputStreamHandlerOptions::ArenaDtor(void* object) {
  DefaultInputStreamHandlerOptions* _this = reinterpret_cast< DefaultInputStreamHandlerOptions* >(object);
  (void)_this;
}
void DefaultInputStreamHandlerOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DefaultInputStreamHandlerOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void DefaultInputStreamHandlerOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.DefaultInputStreamHandlerOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  batch_size_ = 1;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DefaultInputStreamHandlerOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 batch_size = 1 [default = 1];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _Internal::set_has_batch_size(&has_bits);
          batch_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
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

uint8_t* DefaultInputStreamHandlerOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.DefaultInputStreamHandlerOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 batch_size = 1 [default = 1];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_batch_size(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.DefaultInputStreamHandlerOptions)
  return target;
}

size_t DefaultInputStreamHandlerOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.DefaultInputStreamHandlerOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // optional int32 batch_size = 1 [default = 1];
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_batch_size());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData DefaultInputStreamHandlerOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    DefaultInputStreamHandlerOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*DefaultInputStreamHandlerOptions::GetClassData() const { return &_class_data_; }

void DefaultInputStreamHandlerOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<DefaultInputStreamHandlerOptions *>(to)->MergeFrom(
      static_cast<const DefaultInputStreamHandlerOptions &>(from));
}


void DefaultInputStreamHandlerOptions::MergeFrom(const DefaultInputStreamHandlerOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.DefaultInputStreamHandlerOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_batch_size()) {
    _internal_set_batch_size(from._internal_batch_size());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void DefaultInputStreamHandlerOptions::CopyFrom(const DefaultInputStreamHandlerOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.DefaultInputStreamHandlerOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DefaultInputStreamHandlerOptions::IsInitialized() const {
  return true;
}

void DefaultInputStreamHandlerOptions::InternalSwap(DefaultInputStreamHandlerOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(batch_size_, other->batch_size_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DefaultInputStreamHandlerOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2fdefault_5finput_5fstream_5fhandler_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int DefaultInputStreamHandlerOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::MediaPipeOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::DefaultInputStreamHandlerOptions >, 11, false >
  DefaultInputStreamHandlerOptions::ext(kExtFieldNumber, ::mediapipe::DefaultInputStreamHandlerOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::DefaultInputStreamHandlerOptions* Arena::CreateMaybeMessage< ::mediapipe::DefaultInputStreamHandlerOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::DefaultInputStreamHandlerOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
