// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/deps/proto_descriptor.proto

#include "mediapipe/framework/deps/proto_descriptor.pb.h"

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
constexpr FieldDescriptorProto::FieldDescriptorProto(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized){}
struct FieldDescriptorProtoDefaultTypeInternal {
  constexpr FieldDescriptorProtoDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~FieldDescriptorProtoDefaultTypeInternal() {}
  union {
    FieldDescriptorProto _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT FieldDescriptorProtoDefaultTypeInternal _FieldDescriptorProto_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::mediapipe::FieldDescriptorProto, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::mediapipe::FieldDescriptorProto)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_FieldDescriptorProto_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n/mediapipe/framework/deps/proto_descrip"
  "tor.proto\022\tmediapipe\"\341\002\n\024FieldDescriptor"
  "Proto\"\310\002\n\004Type\022\020\n\014TYPE_INVALID\020\000\022\017\n\013TYPE"
  "_DOUBLE\020\001\022\016\n\nTYPE_FLOAT\020\002\022\016\n\nTYPE_INT64\020"
  "\003\022\017\n\013TYPE_UINT64\020\004\022\016\n\nTYPE_INT32\020\005\022\020\n\014TY"
  "PE_FIXED64\020\006\022\020\n\014TYPE_FIXED32\020\007\022\r\n\tTYPE_B"
  "OOL\020\010\022\017\n\013TYPE_STRING\020\t\022\016\n\nTYPE_GROUP\020\n\022\020"
  "\n\014TYPE_MESSAGE\020\013\022\016\n\nTYPE_BYTES\020\014\022\017\n\013TYPE"
  "_UINT32\020\r\022\r\n\tTYPE_ENUM\020\016\022\021\n\rTYPE_SFIXED3"
  "2\020\017\022\021\n\rTYPE_SFIXED64\020\020\022\017\n\013TYPE_SINT32\020\021\022"
  "\017\n\013TYPE_SINT64\020\022B7\n\032com.google.mediapipe"
  ".protoB\031FieldDescriptorProtoProto"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto = {
  false, false, 473, descriptor_table_protodef_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto, "mediapipe/framework/deps/proto_descriptor.proto", 
  &descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto_once, nullptr, 0, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto, file_level_service_descriptors_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto(&descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* FieldDescriptorProto_Type_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto);
  return file_level_enum_descriptors_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto[0];
}
bool FieldDescriptorProto_Type_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_INVALID;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_DOUBLE;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_FLOAT;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_INT64;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_UINT64;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_INT32;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_FIXED64;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_FIXED32;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_BOOL;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_STRING;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_GROUP;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_MESSAGE;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_BYTES;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_UINT32;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_ENUM;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_SFIXED32;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_SFIXED64;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_SINT32;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::TYPE_SINT64;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::Type_MIN;
constexpr FieldDescriptorProto_Type FieldDescriptorProto::Type_MAX;
constexpr int FieldDescriptorProto::Type_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))

// ===================================================================

class FieldDescriptorProto::_Internal {
 public:
};

FieldDescriptorProto::FieldDescriptorProto(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase(arena, is_message_owned) {
  // @@protoc_insertion_point(arena_constructor:mediapipe.FieldDescriptorProto)
}
FieldDescriptorProto::FieldDescriptorProto(const FieldDescriptorProto& from)
  : ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:mediapipe.FieldDescriptorProto)
}





const ::PROTOBUF_NAMESPACE_ID::Message::ClassData FieldDescriptorProto::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::CopyImpl,
    ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::MergeImpl,
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*FieldDescriptorProto::GetClassData() const { return &_class_data_; }







::PROTOBUF_NAMESPACE_ID::Metadata FieldDescriptorProto::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fdeps_2fproto_5fdescriptor_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::FieldDescriptorProto* Arena::CreateMaybeMessage< ::mediapipe::FieldDescriptorProto >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::FieldDescriptorProto >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>