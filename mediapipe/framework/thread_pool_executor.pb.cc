// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/thread_pool_executor.proto

#include "mediapipe/framework/thread_pool_executor.pb.h"

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
constexpr ThreadPoolExecutorOptions::ThreadPoolExecutorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : thread_name_prefix_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , num_threads_(0)
  , stack_size_(0)
  , nice_priority_level_(0)
  , require_processor_performance_(0)
{}
struct ThreadPoolExecutorOptionsDefaultTypeInternal {
  constexpr ThreadPoolExecutorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ThreadPoolExecutorOptionsDefaultTypeInternal() {}
  union {
    ThreadPoolExecutorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ThreadPoolExecutorOptionsDefaultTypeInternal _ThreadPoolExecutorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto = nullptr;

const uint32_t TableStruct_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, num_threads_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, stack_size_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, nice_priority_level_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, require_processor_performance_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::ThreadPoolExecutorOptions, thread_name_prefix_),
  1,
  2,
  3,
  4,
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 11, -1, sizeof(::mediapipe::ThreadPoolExecutorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_ThreadPoolExecutorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n.mediapipe/framework/thread_pool_execut"
  "or.proto\022\tmediapipe\032+mediapipe/framework"
  "/mediapipe_options.proto\"\351\002\n\031ThreadPoolE"
  "xecutorOptions\022\023\n\013num_threads\030\001 \001(\005\022\022\n\ns"
  "tack_size\030\002 \001(\005\022\033\n\023nice_priority_level\030\003"
  " \001(\005\022`\n\035require_processor_performance\030\004 "
  "\001(\01629.mediapipe.ThreadPoolExecutorOption"
  "s.ProcessorPerformance\022\032\n\022thread_name_pr"
  "efix\030\005 \001(\t\"5\n\024ProcessorPerformance\022\n\n\006NO"
  "RMAL\020\000\022\007\n\003LOW\020\001\022\010\n\004HIGH\020\0022Q\n\003ext\022\033.media"
  "pipe.MediaPipeOptions\030\223\323\365J \001(\0132$.mediapi"
  "pe.ThreadPoolExecutorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fmediapipe_5foptions_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto = {
  false, false, 468, descriptor_table_protodef_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto, "mediapipe/framework/thread_pool_executor.proto", 
  &descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_once, descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto, file_level_service_descriptors_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto(&descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* ThreadPoolExecutorOptions_ProcessorPerformance_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto);
  return file_level_enum_descriptors_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto[0];
}
bool ThreadPoolExecutorOptions_ProcessorPerformance_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))
constexpr ThreadPoolExecutorOptions_ProcessorPerformance ThreadPoolExecutorOptions::NORMAL;
constexpr ThreadPoolExecutorOptions_ProcessorPerformance ThreadPoolExecutorOptions::LOW;
constexpr ThreadPoolExecutorOptions_ProcessorPerformance ThreadPoolExecutorOptions::HIGH;
constexpr ThreadPoolExecutorOptions_ProcessorPerformance ThreadPoolExecutorOptions::ProcessorPerformance_MIN;
constexpr ThreadPoolExecutorOptions_ProcessorPerformance ThreadPoolExecutorOptions::ProcessorPerformance_MAX;
constexpr int ThreadPoolExecutorOptions::ProcessorPerformance_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))

// ===================================================================

class ThreadPoolExecutorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<ThreadPoolExecutorOptions>()._has_bits_);
  static void set_has_num_threads(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_stack_size(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_nice_priority_level(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_require_processor_performance(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_thread_name_prefix(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

ThreadPoolExecutorOptions::ThreadPoolExecutorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:mediapipe.ThreadPoolExecutorOptions)
}
ThreadPoolExecutorOptions::ThreadPoolExecutorOptions(const ThreadPoolExecutorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  thread_name_prefix_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    thread_name_prefix_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (from._internal_has_thread_name_prefix()) {
    thread_name_prefix_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_thread_name_prefix(), 
      GetArenaForAllocation());
  }
  ::memcpy(&num_threads_, &from.num_threads_,
    static_cast<size_t>(reinterpret_cast<char*>(&require_processor_performance_) -
    reinterpret_cast<char*>(&num_threads_)) + sizeof(require_processor_performance_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.ThreadPoolExecutorOptions)
}

inline void ThreadPoolExecutorOptions::SharedCtor() {
thread_name_prefix_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  thread_name_prefix_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&num_threads_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&require_processor_performance_) -
    reinterpret_cast<char*>(&num_threads_)) + sizeof(require_processor_performance_));
}

ThreadPoolExecutorOptions::~ThreadPoolExecutorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.ThreadPoolExecutorOptions)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void ThreadPoolExecutorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  thread_name_prefix_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void ThreadPoolExecutorOptions::ArenaDtor(void* object) {
  ThreadPoolExecutorOptions* _this = reinterpret_cast< ThreadPoolExecutorOptions* >(object);
  (void)_this;
}
void ThreadPoolExecutorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ThreadPoolExecutorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void ThreadPoolExecutorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.ThreadPoolExecutorOptions)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    thread_name_prefix_.ClearNonDefaultToEmpty();
  }
  if (cached_has_bits & 0x0000001eu) {
    ::memset(&num_threads_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&require_processor_performance_) -
        reinterpret_cast<char*>(&num_threads_)) + sizeof(require_processor_performance_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ThreadPoolExecutorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 num_threads = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _Internal::set_has_num_threads(&has_bits);
          num_threads_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 stack_size = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 16)) {
          _Internal::set_has_stack_size(&has_bits);
          stack_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional int32 nice_priority_level = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 24)) {
          _Internal::set_has_nice_priority_level(&has_bits);
          nice_priority_level_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // optional .mediapipe.ThreadPoolExecutorOptions.ProcessorPerformance require_processor_performance = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 32)) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::ThreadPoolExecutorOptions_ProcessorPerformance_IsValid(val))) {
            _internal_set_require_processor_performance(static_cast<::mediapipe::ThreadPoolExecutorOptions_ProcessorPerformance>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(4, val, mutable_unknown_fields());
          }
        } else
          goto handle_unusual;
        continue;
      // optional string thread_name_prefix = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 42)) {
          auto str = _internal_mutable_thread_name_prefix();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.ThreadPoolExecutorOptions.thread_name_prefix");
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

uint8_t* ThreadPoolExecutorOptions::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.ThreadPoolExecutorOptions)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 num_threads = 1;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_num_threads(), target);
  }

  // optional int32 stack_size = 2;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_stack_size(), target);
  }

  // optional int32 nice_priority_level = 3;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_nice_priority_level(), target);
  }

  // optional .mediapipe.ThreadPoolExecutorOptions.ProcessorPerformance require_processor_performance = 4;
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      4, this->_internal_require_processor_performance(), target);
  }

  // optional string thread_name_prefix = 5;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_thread_name_prefix().data(), static_cast<int>(this->_internal_thread_name_prefix().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.ThreadPoolExecutorOptions.thread_name_prefix");
    target = stream->WriteStringMaybeAliased(
        5, this->_internal_thread_name_prefix(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.ThreadPoolExecutorOptions)
  return target;
}

size_t ThreadPoolExecutorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.ThreadPoolExecutorOptions)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000001fu) {
    // optional string thread_name_prefix = 5;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_thread_name_prefix());
    }

    // optional int32 num_threads = 1;
    if (cached_has_bits & 0x00000002u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_num_threads());
    }

    // optional int32 stack_size = 2;
    if (cached_has_bits & 0x00000004u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_stack_size());
    }

    // optional int32 nice_priority_level = 3;
    if (cached_has_bits & 0x00000008u) {
      total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32SizePlusOne(this->_internal_nice_priority_level());
    }

    // optional .mediapipe.ThreadPoolExecutorOptions.ProcessorPerformance require_processor_performance = 4;
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_require_processor_performance());
    }

  }
  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ThreadPoolExecutorOptions::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    ThreadPoolExecutorOptions::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ThreadPoolExecutorOptions::GetClassData() const { return &_class_data_; }

void ThreadPoolExecutorOptions::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<ThreadPoolExecutorOptions *>(to)->MergeFrom(
      static_cast<const ThreadPoolExecutorOptions &>(from));
}


void ThreadPoolExecutorOptions::MergeFrom(const ThreadPoolExecutorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.ThreadPoolExecutorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000001fu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_thread_name_prefix(from._internal_thread_name_prefix());
    }
    if (cached_has_bits & 0x00000002u) {
      num_threads_ = from.num_threads_;
    }
    if (cached_has_bits & 0x00000004u) {
      stack_size_ = from.stack_size_;
    }
    if (cached_has_bits & 0x00000008u) {
      nice_priority_level_ = from.nice_priority_level_;
    }
    if (cached_has_bits & 0x00000010u) {
      require_processor_performance_ = from.require_processor_performance_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ThreadPoolExecutorOptions::CopyFrom(const ThreadPoolExecutorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.ThreadPoolExecutorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ThreadPoolExecutorOptions::IsInitialized() const {
  return true;
}

void ThreadPoolExecutorOptions::InternalSwap(ThreadPoolExecutorOptions* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &thread_name_prefix_, lhs_arena,
      &other->thread_name_prefix_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(ThreadPoolExecutorOptions, require_processor_performance_)
      + sizeof(ThreadPoolExecutorOptions::require_processor_performance_)
      - PROTOBUF_FIELD_OFFSET(ThreadPoolExecutorOptions, num_threads_)>(
          reinterpret_cast<char*>(&num_threads_),
          reinterpret_cast<char*>(&other->num_threads_));
}

::PROTOBUF_NAMESPACE_ID::Metadata ThreadPoolExecutorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fthread_5fpool_5fexecutor_2eproto[0]);
}
#if !defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912)
const int ThreadPoolExecutorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::MediaPipeOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::ThreadPoolExecutorOptions >, 11, false >
  ThreadPoolExecutorOptions::ext(kExtFieldNumber, ::mediapipe::ThreadPoolExecutorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::ThreadPoolExecutorOptions* Arena::CreateMaybeMessage< ::mediapipe::ThreadPoolExecutorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::ThreadPoolExecutorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>