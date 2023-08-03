// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/tool/switch_container.proto

#include "mediapipe/framework/tool/switch_container.pb.h"

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
constexpr SwitchContainerOptions::SwitchContainerOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : contained_node_()
  , select_(0)
  , enable_(false)
  , synchronize_io_(false){}
struct SwitchContainerOptionsDefaultTypeInternal {
  constexpr SwitchContainerOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SwitchContainerOptionsDefaultTypeInternal() {}
  union {
    SwitchContainerOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SwitchContainerOptionsDefaultTypeInternal _SwitchContainerOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::SwitchContainerOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SwitchContainerOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::SwitchContainerOptions, contained_node_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SwitchContainerOptions, select_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SwitchContainerOptions, enable_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SwitchContainerOptions, synchronize_io_),
  ~0u,
  0,
  1,
  2,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 10, -1, sizeof(::mediapipe::SwitchContainerOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_SwitchContainerOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n/mediapipe/framework/tool/switch_contai"
  "ner.proto\022\tmediapipe\032$mediapipe/framewor"
  "k/calculator.proto\"\347\001\n\026SwitchContainerOp"
  "tions\022=\n\016contained_node\030\002 \003(\0132%.mediapip"
  "e.CalculatorGraphConfig.Node\022\016\n\006select\030\003"
  " \001(\005\022\016\n\006enable\030\004 \001(\010\022\026\n\016synchronize_io\030\005"
  " \001(\0102P\n\003ext\022\034.mediapipe.CalculatorOption"
  "s\030\342\232\374\244\001 \001(\0132!.mediapipe.SwitchContainerO"
  "ptionsJ\004\010\001\020\002B2\n\032com.google.mediapipe.pro"
  "toB\024SwitchContainerProto"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto = {
  false, false, 384, descriptor_table_protodef_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto, "mediapipe/framework/tool/switch_container.proto", 
  &descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_once, descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto, file_level_service_descriptors_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto(&descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto);
namespace mediapipe {

// ===================================================================

class SwitchContainerOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<SwitchContainerOptions>()._has_bits_);
  static void set_has_select(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_enable(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_synchronize_io(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
};

void SwitchContainerOptions::clear_contained_node() {
  contained_node_.Clear();
}
SwitchContainerOptions::SwitchContainerOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  contained_node_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.SwitchContainerOptions)
}
SwitchContainerOptions::SwitchContainerOptions(const SwitchContainerOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      contained_node_(from.contained_node_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&select_, &from.select_,
    static_cast<size_t>(reinterpret_cast<char*>(&synchronize_io_) -
    reinterpret_cast<char*>(&select_)) + sizeof(synchronize_io_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.SwitchContainerOptions)
}

inline void SwitchContainerOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&select_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&synchronize_io_) -
    reinterpret_cast<char*>(&select_)) + sizeof(synchronize_io_));
}

SwitchContainerOptions::~SwitchContainerOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.SwitchContainerOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void SwitchContainerOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void SwitchContainerOptions::ArenaDtor(void* object) {
  SwitchContainerOptions* _this = reinterpret_cast< SwitchContainerOptions* >(object);
  (void)_this;
}
void SwitchContainerOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SwitchContainerOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SwitchContainerOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.SwitchContainerOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  contained_node_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    ::memset(&select_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&synchronize_io_) -
        reinterpret_cast<char*>(&select_)) + sizeof(synchronize_io_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SwitchContainerOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .mediapipe.CalculatorGraphConfig.Node contained_node = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_contained_node(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
        } else
          goto handle_unusual;
        continue;
      // optional int32 select = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 24)) {
          _Internal::set_has_select(&has_bits);
          select_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional bool enable = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 32)) {
          _Internal::set_has_enable(&has_bits);
          enable_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional bool synchronize_io = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 40)) {
          _Internal::set_has_synchronize_io(&has_bits);
          synchronize_io_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

uint8_t* SwitchContainerOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.SwitchContainerOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .mediapipe.CalculatorGraphConfig.Node contained_node = 2;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_contained_node_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(2, this->_internal_contained_node(i), target, stream);
  }

  cached_has_bits = _has_bits_[0];
  // optional int32 select = 3;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_select(), target);
  }

  // optional bool enable = 4;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(4, this->_internal_enable(), target);
  }

  // optional bool synchronize_io = 5;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(5, this->_internal_synchronize_io(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.SwitchContainerOptions)
  return target;
}

size_t SwitchContainerOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.SwitchContainerOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .mediapipe.CalculatorGraphConfig.Node contained_node = 2;
  total_size += 1UL * this->_internal_contained_node_size();
  for (const auto& msg : this->contained_node_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    // optional int32 select = 3;
    if (cached_has_bits & 0x00000001u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_select());
    }

    // optional bool enable = 4;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 + 1;
    }

    // optional bool synchronize_io = 5;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 + 1;
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SwitchContainerOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    SwitchContainerOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SwitchContainerOptions::GetClassData() const { return &_class_data_; }

void SwitchContainerOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<SwitchContainerOptions *>(to)->MergeFrom(
      static_cast<const SwitchContainerOptions &>(from));
}


void SwitchContainerOptions::MergeFrom(const SwitchContainerOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.SwitchContainerOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  contained_node_.MergeFrom(from.contained_node_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      select_ = from.select_;
    }
    if (cached_has_bits & 0x00000002u) {
      enable_ = from.enable_;
    }
    if (cached_has_bits & 0x00000004u) {
      synchronize_io_ = from.synchronize_io_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SwitchContainerOptions::CopyFrom(const SwitchContainerOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.SwitchContainerOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SwitchContainerOptions::IsInitialized() const {
  if (!::PROTOBUF_NAMESPACE_ID::internal::AllAreInitialized(contained_node_))
    return false;
  return true;
}

void SwitchContainerOptions::InternalSwap(SwitchContainerOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  contained_node_.InternalSwap(&other->contained_node_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(SwitchContainerOptions, synchronize_io_)
      + sizeof(SwitchContainerOptions::synchronize_io_)
      - PROTOBUF_FIELD_OFFSET(SwitchContainerOptions, select_)>(
          reinterpret_cast<char*>(&select_),
          reinterpret_cast<char*>(&other->select_));
}

::PROTOBUF_NAMESPACE_ID::Metadata SwitchContainerOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_getter, &descriptor_table_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2ftool_2fswitch_5fcontainer_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int SwitchContainerOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::SwitchContainerOptions >, 11, false >
  SwitchContainerOptions::ext(kExtFieldNumber, ::mediapipe::SwitchContainerOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::SwitchContainerOptions* Arena::CreateMaybeMessage< ::mediapipe::SwitchContainerOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::SwitchContainerOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
