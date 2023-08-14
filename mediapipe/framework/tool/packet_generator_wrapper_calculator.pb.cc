// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/tool/packet_generator_wrapper_calculator.proto

#include "mediapipe/framework/tool/packet_generator_wrapper_calculator.pb.h"

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
constexpr PacketGeneratorWrapperCalculatorOptions::PacketGeneratorWrapperCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : packet_generator_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , package_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , options_(nullptr){}
struct PacketGeneratorWrapperCalculatorOptionsDefaultTypeInternal {
  constexpr PacketGeneratorWrapperCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~PacketGeneratorWrapperCalculatorOptionsDefaultTypeInternal() {}
  union {
    PacketGeneratorWrapperCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PacketGeneratorWrapperCalculatorOptionsDefaultTypeInternal _PacketGeneratorWrapperCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::PacketGeneratorWrapperCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::PacketGeneratorWrapperCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::PacketGeneratorWrapperCalculatorOptions, packet_generator_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::PacketGeneratorWrapperCalculatorOptions, options_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::PacketGeneratorWrapperCalculatorOptions, package_),
  0,
  2,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 9, -1, sizeof(::mediapipe::PacketGeneratorWrapperCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_PacketGeneratorWrapperCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nBmediapipe/framework/tool/packet_genera"
  "tor_wrapper_calculator.proto\022\tmediapipe\032"
  ",mediapipe/framework/calculator_options."
  "proto\032*mediapipe/framework/packet_genera"
  "tor.proto\"\353\001\n\'PacketGeneratorWrapperCalc"
  "ulatorOptions\022\030\n\020packet_generator\030\001 \001(\t\022"
  "2\n\007options\030\002 \001(\0132!.mediapipe.PacketGener"
  "atorOptions\022\017\n\007package\030\003 \001(\t2a\n\003ext\022\034.me"
  "diapipe.CalculatorOptions\030\345\214\220\266\001 \001(\01322.me"
  "diapipe.PacketGeneratorWrapperCalculator"
  "Options"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_5foptions_2eproto,
  &::descriptor_table_mediapipe_2fframework_2fpacket_5fgenerator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto = {
  false, false, 407, descriptor_table_protodef_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto, "mediapipe/framework/tool/packet_generator_wrapper_calculator.proto", 
  &descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto(&descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class PacketGeneratorWrapperCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<PacketGeneratorWrapperCalculatorOptions>()._has_bits_);
  static void set_has_packet_generator(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::mediapipe::PacketGeneratorOptions& options(const PacketGeneratorWrapperCalculatorOptions* msg);
  static void set_has_options(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_package(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::mediapipe::PacketGeneratorOptions&
PacketGeneratorWrapperCalculatorOptions::_Internal::options(const PacketGeneratorWrapperCalculatorOptions* msg) {
  return *msg->options_;
}
void PacketGeneratorWrapperCalculatorOptions::clear_options() {
  if (options_ != nullptr) options_->Clear();
  _has_bits_[0] &= ~0x00000004u;
}
PacketGeneratorWrapperCalculatorOptions::PacketGeneratorWrapperCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.PacketGeneratorWrapperCalculatorOptions)
}
PacketGeneratorWrapperCalculatorOptions::PacketGeneratorWrapperCalculatorOptions(const PacketGeneratorWrapperCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  packet_generator_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    packet_generator_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (from._internal_has_packet_generator()) {
    packet_generator_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_packet_generator(), 
      GetArenaForAllocation());
  }
  package_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    package_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (from._internal_has_package()) {
    package_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_package(), 
      GetArenaForAllocation());
  }
  if (from._internal_has_options()) {
    options_ = new ::mediapipe::PacketGeneratorOptions(*from.options_);
  } else {
    options_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:mediapipe.PacketGeneratorWrapperCalculatorOptions)
}

inline void PacketGeneratorWrapperCalculatorOptions::SharedCtor() {
packet_generator_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  packet_generator_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
package_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  package_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
options_ = nullptr;
}

PacketGeneratorWrapperCalculatorOptions::~PacketGeneratorWrapperCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void PacketGeneratorWrapperCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  packet_generator_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  package_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (this != internal_default_instance()) delete options_;
}

void PacketGeneratorWrapperCalculatorOptions::ArenaDtor(void* object) {
  PacketGeneratorWrapperCalculatorOptions* _this = reinterpret_cast< PacketGeneratorWrapperCalculatorOptions* >(object);
  (void)_this;
}
void PacketGeneratorWrapperCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void PacketGeneratorWrapperCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void PacketGeneratorWrapperCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      packet_generator_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000002u) {
      package_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000004u) {
      GOOGLE_DCHECK(options_ != nullptr);
      options_->Clear();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* PacketGeneratorWrapperCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional string packet_generator = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_packet_generator();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.PacketGeneratorWrapperCalculatorOptions.packet_generator");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.PacketGeneratorOptions options = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_options(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional string package = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          auto str = _internal_mutable_package();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.PacketGeneratorWrapperCalculatorOptions.package");
          #endif  // !NDEBUG
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

uint8_t* PacketGeneratorWrapperCalculatorOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string packet_generator = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_packet_generator().data(), static_cast<int>(this->_internal_packet_generator().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.PacketGeneratorWrapperCalculatorOptions.packet_generator");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_packet_generator(), target);
  }

  // optional .mediapipe.PacketGeneratorOptions options = 2;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        2, _Internal::options(this), target, stream);
  }

  // optional string package = 3;
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_package().data(), static_cast<int>(this->_internal_package().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.PacketGeneratorWrapperCalculatorOptions.package");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_package(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  return target;
}

size_t PacketGeneratorWrapperCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    // optional string packet_generator = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_packet_generator());
    }

    // optional string package = 3;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_package());
    }

    // optional .mediapipe.PacketGeneratorOptions options = 2;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *options_);
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData PacketGeneratorWrapperCalculatorOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    PacketGeneratorWrapperCalculatorOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*PacketGeneratorWrapperCalculatorOptions::GetClassData() const { return &_class_data_; }

void PacketGeneratorWrapperCalculatorOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<PacketGeneratorWrapperCalculatorOptions *>(to)->MergeFrom(
      static_cast<const PacketGeneratorWrapperCalculatorOptions &>(from));
}


void PacketGeneratorWrapperCalculatorOptions::MergeFrom(const PacketGeneratorWrapperCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_packet_generator(from._internal_packet_generator());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_set_package(from._internal_package());
    }
    if (cached_has_bits & 0x00000004u) {
      _internal_mutable_options()->::mediapipe::PacketGeneratorOptions::MergeFrom(from._internal_options());
    }
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void PacketGeneratorWrapperCalculatorOptions::CopyFrom(const PacketGeneratorWrapperCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.PacketGeneratorWrapperCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool PacketGeneratorWrapperCalculatorOptions::IsInitialized() const {
  if (_internal_has_options()) {
    if (!options_->IsInitialized()) return false;
  }
  return true;
}

void PacketGeneratorWrapperCalculatorOptions::InternalSwap(PacketGeneratorWrapperCalculatorOptions* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &packet_generator_, lhs_arena,
      &other->packet_generator_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &package_, lhs_arena,
      &other->package_, rhs_arena
  );
  swap(options_, other->options_);
}

::PROTOBUF_NAMESPACE_ID::Metadata PacketGeneratorWrapperCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2ftool_2fpacket_5fgenerator_5fwrapper_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int PacketGeneratorWrapperCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::PacketGeneratorWrapperCalculatorOptions >, 11, false >
  PacketGeneratorWrapperCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::PacketGeneratorWrapperCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::PacketGeneratorWrapperCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::PacketGeneratorWrapperCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::PacketGeneratorWrapperCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>