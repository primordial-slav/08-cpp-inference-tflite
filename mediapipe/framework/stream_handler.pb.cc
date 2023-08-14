// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/stream_handler.proto

#include "mediapipe/framework/stream_handler.pb.h"

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
constexpr InputStreamHandlerConfig::InputStreamHandlerConfig(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : input_stream_handler_(nullptr)
  , options_(nullptr){}
struct InputStreamHandlerConfigDefaultTypeInternal {
  constexpr InputStreamHandlerConfigDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~InputStreamHandlerConfigDefaultTypeInternal() {}
  union {
    InputStreamHandlerConfig _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT InputStreamHandlerConfigDefaultTypeInternal _InputStreamHandlerConfig_default_instance_;
constexpr OutputStreamHandlerConfig::OutputStreamHandlerConfig(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : input_side_packet_()
  , output_stream_handler_(nullptr)
  , options_(nullptr){}
struct OutputStreamHandlerConfigDefaultTypeInternal {
  constexpr OutputStreamHandlerConfigDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~OutputStreamHandlerConfigDefaultTypeInternal() {}
  union {
    OutputStreamHandlerConfig _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT OutputStreamHandlerConfigDefaultTypeInternal _OutputStreamHandlerConfig_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fframework_2fstream_5fhandler_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2fstream_5fhandler_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fframework_2fstream_5fhandler_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::InputStreamHandlerConfig, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::InputStreamHandlerConfig, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::InputStreamHandlerConfig, input_stream_handler_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::InputStreamHandlerConfig, options_),
  0,
  1,
  PROTOBUF_FIELD_OFFSET(::mediapipe::OutputStreamHandlerConfig, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::OutputStreamHandlerConfig, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::OutputStreamHandlerConfig, output_stream_handler_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::OutputStreamHandlerConfig, input_side_packet_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::OutputStreamHandlerConfig, options_),
  0,
  ~0u,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, -1, sizeof(::mediapipe::InputStreamHandlerConfig)},
  { 10, 19, -1, sizeof(::mediapipe::OutputStreamHandlerConfig)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_InputStreamHandlerConfig_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_OutputStreamHandlerConfig_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2fstream_5fhandler_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n(mediapipe/framework/stream_handler.pro"
  "to\022\tmediapipe\032+mediapipe/framework/media"
  "pipe_options.proto\"\201\001\n\030InputStreamHandle"
  "rConfig\0227\n\024input_stream_handler\030\001 \001(\t:\031D"
  "efaultInputStreamHandler\022,\n\007options\030\003 \001("
  "\0132\033.mediapipe.MediaPipeOptions\"\237\001\n\031Outpu"
  "tStreamHandlerConfig\0229\n\025output_stream_ha"
  "ndler\030\001 \001(\t:\032InOrderOutputStreamHandler\022"
  "\031\n\021input_side_packet\030\002 \003(\t\022,\n\007options\030\003 "
  "\001(\0132\033.mediapipe.MediaPipeOptionsB0\n\032com."
  "google.mediapipe.protoB\022StreamHandlerPro"
  "to"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fmediapipe_5foptions_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto = {
  false, false, 442, descriptor_table_protodef_mediapipe_2fframework_2fstream_5fhandler_2eproto, "mediapipe/framework/stream_handler.proto", 
  &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_once, descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_deps, 1, 2,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2fstream_5fhandler_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2fstream_5fhandler_2eproto, file_level_service_descriptors_mediapipe_2fframework_2fstream_5fhandler_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2fstream_5fhandler_2eproto(&descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto);
namespace mediapipe {

// ===================================================================

class InputStreamHandlerConfig::_Internal {
 public:
  using HasBits = decltype(std::declval<InputStreamHandlerConfig>()._has_bits_);
  static void set_has_input_stream_handler(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::mediapipe::MediaPipeOptions& options(const InputStreamHandlerConfig* msg);
  static void set_has_options(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::mediapipe::MediaPipeOptions&
InputStreamHandlerConfig::_Internal::options(const InputStreamHandlerConfig* msg) {
  return *msg->options_;
}
const ::PROTOBUF_NAMESPACE_ID::internal::LazyString InputStreamHandlerConfig::_i_give_permission_to_break_this_code_default_input_stream_handler_{{{"DefaultInputStreamHandler", 25}}, {nullptr}};
void InputStreamHandlerConfig::clear_options() {
  if (options_ != nullptr) options_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
InputStreamHandlerConfig::InputStreamHandlerConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.InputStreamHandlerConfig)
}
InputStreamHandlerConfig::InputStreamHandlerConfig(const InputStreamHandlerConfig& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  input_stream_handler_.UnsafeSetDefault(nullptr);
  if (from._internal_has_input_stream_handler()) {
    input_stream_handler_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, from._internal_input_stream_handler(), 
      GetArenaForAllocation());
  }
  if (from._internal_has_options()) {
    options_ = new ::mediapipe::MediaPipeOptions(*from.options_);
  } else {
    options_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:mediapipe.InputStreamHandlerConfig)
}

inline void InputStreamHandlerConfig::SharedCtor() {
input_stream_handler_.UnsafeSetDefault(nullptr);
options_ = nullptr;
}

InputStreamHandlerConfig::~InputStreamHandlerConfig() {
  // @@protoc_insertion_point(destructor:mediapipe.InputStreamHandlerConfig)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void InputStreamHandlerConfig::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  input_stream_handler_.DestroyNoArena(nullptr);
  if (this != internal_default_instance()) delete options_;
}

void InputStreamHandlerConfig::ArenaDtor(void* object) {
  InputStreamHandlerConfig* _this = reinterpret_cast< InputStreamHandlerConfig* >(object);
  (void)_this;
}
void InputStreamHandlerConfig::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void InputStreamHandlerConfig::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void InputStreamHandlerConfig::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.InputStreamHandlerConfig)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      input_stream_handler_.ClearToDefault(::mediapipe::InputStreamHandlerConfig::_i_give_permission_to_break_this_code_default_input_stream_handler_, GetArenaForAllocation());
       }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(options_ != nullptr);
      options_->Clear();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* InputStreamHandlerConfig::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional string input_stream_handler = 1 [default = "DefaultInputStreamHandler"];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_input_stream_handler();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.InputStreamHandlerConfig.input_stream_handler");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.MediaPipeOptions options = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_options(), ptr);
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

uint8_t* InputStreamHandlerConfig::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.InputStreamHandlerConfig)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string input_stream_handler = 1 [default = "DefaultInputStreamHandler"];
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_input_stream_handler().data(), static_cast<int>(this->_internal_input_stream_handler().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.InputStreamHandlerConfig.input_stream_handler");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_input_stream_handler(), target);
  }

  // optional .mediapipe.MediaPipeOptions options = 3;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        3, _Internal::options(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.InputStreamHandlerConfig)
  return target;
}

size_t InputStreamHandlerConfig::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.InputStreamHandlerConfig)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional string input_stream_handler = 1 [default = "DefaultInputStreamHandler"];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_input_stream_handler());
    }

    // optional .mediapipe.MediaPipeOptions options = 3;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *options_);
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData InputStreamHandlerConfig::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    InputStreamHandlerConfig::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*InputStreamHandlerConfig::GetClassData() const { return &_class_data_; }

void InputStreamHandlerConfig::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<InputStreamHandlerConfig *>(to)->MergeFrom(
      static_cast<const InputStreamHandlerConfig &>(from));
}


void InputStreamHandlerConfig::MergeFrom(const InputStreamHandlerConfig& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.InputStreamHandlerConfig)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_input_stream_handler(from._internal_input_stream_handler());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_options()->::mediapipe::MediaPipeOptions::MergeFrom(from._internal_options());
    }
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void InputStreamHandlerConfig::CopyFrom(const InputStreamHandlerConfig& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.InputStreamHandlerConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool InputStreamHandlerConfig::IsInitialized() const {
  if (_internal_has_options()) {
    if (!options_->IsInitialized()) return false;
  }
  return true;
}

void InputStreamHandlerConfig::InternalSwap(InputStreamHandlerConfig* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      nullptr,
      &input_stream_handler_, lhs_arena,
      &other->input_stream_handler_, rhs_arena
  );
  swap(options_, other->options_);
}

::PROTOBUF_NAMESPACE_ID::Metadata InputStreamHandlerConfig::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2eproto[0]);
}

// ===================================================================

class OutputStreamHandlerConfig::_Internal {
 public:
  using HasBits = decltype(std::declval<OutputStreamHandlerConfig>()._has_bits_);
  static void set_has_output_stream_handler(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::mediapipe::MediaPipeOptions& options(const OutputStreamHandlerConfig* msg);
  static void set_has_options(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::mediapipe::MediaPipeOptions&
OutputStreamHandlerConfig::_Internal::options(const OutputStreamHandlerConfig* msg) {
  return *msg->options_;
}
const ::PROTOBUF_NAMESPACE_ID::internal::LazyString OutputStreamHandlerConfig::_i_give_permission_to_break_this_code_default_output_stream_handler_{{{"InOrderOutputStreamHandler", 26}}, {nullptr}};
void OutputStreamHandlerConfig::clear_options() {
  if (options_ != nullptr) options_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
OutputStreamHandlerConfig::OutputStreamHandlerConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  input_side_packet_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.OutputStreamHandlerConfig)
}
OutputStreamHandlerConfig::OutputStreamHandlerConfig(const OutputStreamHandlerConfig& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      input_side_packet_(from.input_side_packet_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  output_stream_handler_.UnsafeSetDefault(nullptr);
  if (from._internal_has_output_stream_handler()) {
    output_stream_handler_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, from._internal_output_stream_handler(), 
      GetArenaForAllocation());
  }
  if (from._internal_has_options()) {
    options_ = new ::mediapipe::MediaPipeOptions(*from.options_);
  } else {
    options_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:mediapipe.OutputStreamHandlerConfig)
}

inline void OutputStreamHandlerConfig::SharedCtor() {
output_stream_handler_.UnsafeSetDefault(nullptr);
options_ = nullptr;
}

OutputStreamHandlerConfig::~OutputStreamHandlerConfig() {
  // @@protoc_insertion_point(destructor:mediapipe.OutputStreamHandlerConfig)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void OutputStreamHandlerConfig::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  output_stream_handler_.DestroyNoArena(nullptr);
  if (this != internal_default_instance()) delete options_;
}

void OutputStreamHandlerConfig::ArenaDtor(void* object) {
  OutputStreamHandlerConfig* _this = reinterpret_cast< OutputStreamHandlerConfig* >(object);
  (void)_this;
}
void OutputStreamHandlerConfig::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void OutputStreamHandlerConfig::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void OutputStreamHandlerConfig::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.OutputStreamHandlerConfig)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  input_side_packet_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      output_stream_handler_.ClearToDefault(::mediapipe::OutputStreamHandlerConfig::_i_give_permission_to_break_this_code_default_output_stream_handler_, GetArenaForAllocation());
       }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(options_ != nullptr);
      options_->Clear();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* OutputStreamHandlerConfig::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional string output_stream_handler = 1 [default = "InOrderOutputStreamHandler"];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_output_stream_handler();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.OutputStreamHandlerConfig.output_stream_handler");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated string input_side_packet = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            auto str = _internal_add_input_side_packet();
            ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
            #ifndef NDEBUG
            ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.OutputStreamHandlerConfig.input_side_packet");
            #endif  // !NDEBUG
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.MediaPipeOptions options = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_options(), ptr);
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

uint8_t* OutputStreamHandlerConfig::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.OutputStreamHandlerConfig)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string output_stream_handler = 1 [default = "InOrderOutputStreamHandler"];
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_output_stream_handler().data(), static_cast<int>(this->_internal_output_stream_handler().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.OutputStreamHandlerConfig.output_stream_handler");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_output_stream_handler(), target);
  }

  // repeated string input_side_packet = 2;
  for (int i = 0, n = this->_internal_input_side_packet_size(); i < n; i++) {
    const auto& s = this->_internal_input_side_packet(i);
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      s.data(), static_cast<int>(s.length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.OutputStreamHandlerConfig.input_side_packet");
    target = stream->WriteString(2, s, target);
  }

  // optional .mediapipe.MediaPipeOptions options = 3;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        3, _Internal::options(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.OutputStreamHandlerConfig)
  return target;
}

size_t OutputStreamHandlerConfig::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.OutputStreamHandlerConfig)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated string input_side_packet = 2;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(input_side_packet_.size());
  for (int i = 0, n = input_side_packet_.size(); i < n; i++) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
      input_side_packet_.Get(i));
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional string output_stream_handler = 1 [default = "InOrderOutputStreamHandler"];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_output_stream_handler());
    }

    // optional .mediapipe.MediaPipeOptions options = 3;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *options_);
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData OutputStreamHandlerConfig::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    OutputStreamHandlerConfig::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*OutputStreamHandlerConfig::GetClassData() const { return &_class_data_; }

void OutputStreamHandlerConfig::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<OutputStreamHandlerConfig *>(to)->MergeFrom(
      static_cast<const OutputStreamHandlerConfig &>(from));
}


void OutputStreamHandlerConfig::MergeFrom(const OutputStreamHandlerConfig& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.OutputStreamHandlerConfig)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  input_side_packet_.MergeFrom(from.input_side_packet_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_output_stream_handler(from._internal_output_stream_handler());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_options()->::mediapipe::MediaPipeOptions::MergeFrom(from._internal_options());
    }
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void OutputStreamHandlerConfig::CopyFrom(const OutputStreamHandlerConfig& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.OutputStreamHandlerConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool OutputStreamHandlerConfig::IsInitialized() const {
  if (_internal_has_options()) {
    if (!options_->IsInitialized()) return false;
  }
  return true;
}

void OutputStreamHandlerConfig::InternalSwap(OutputStreamHandlerConfig* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  input_side_packet_.InternalSwap(&other->input_side_packet_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      nullptr,
      &output_stream_handler_, lhs_arena,
      &other->output_stream_handler_, rhs_arena
  );
  swap(options_, other->options_);
}

::PROTOBUF_NAMESPACE_ID::Metadata OutputStreamHandlerConfig::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fstream_5fhandler_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::InputStreamHandlerConfig* Arena::CreateMaybeMessage< ::mediapipe::InputStreamHandlerConfig >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::InputStreamHandlerConfig >(arena);
}
template<> PROTOBUF_NOINLINE ::mediapipe::OutputStreamHandlerConfig* Arena::CreateMaybeMessage< ::mediapipe::OutputStreamHandlerConfig >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::OutputStreamHandlerConfig >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>