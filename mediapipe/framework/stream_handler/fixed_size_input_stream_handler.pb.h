// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/stream_handler/fixed_size_input_stream_handler.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3019000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3019001 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/mediapipe_options.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto;
namespace mediapipe {
class FixedSizeInputStreamHandlerOptions;
struct FixedSizeInputStreamHandlerOptionsDefaultTypeInternal;
extern FixedSizeInputStreamHandlerOptionsDefaultTypeInternal _FixedSizeInputStreamHandlerOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::FixedSizeInputStreamHandlerOptions* Arena::CreateMaybeMessage<::mediapipe::FixedSizeInputStreamHandlerOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class FixedSizeInputStreamHandlerOptions final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.FixedSizeInputStreamHandlerOptions) */ {
 public:
  inline FixedSizeInputStreamHandlerOptions() : FixedSizeInputStreamHandlerOptions(nullptr) {}
  ~FixedSizeInputStreamHandlerOptions() override;
  explicit constexpr FixedSizeInputStreamHandlerOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  FixedSizeInputStreamHandlerOptions(const FixedSizeInputStreamHandlerOptions& from);
  FixedSizeInputStreamHandlerOptions(FixedSizeInputStreamHandlerOptions&& from) noexcept
    : FixedSizeInputStreamHandlerOptions() {
    *this = ::std::move(from);
  }

  inline FixedSizeInputStreamHandlerOptions& operator=(const FixedSizeInputStreamHandlerOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline FixedSizeInputStreamHandlerOptions& operator=(FixedSizeInputStreamHandlerOptions&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const FixedSizeInputStreamHandlerOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const FixedSizeInputStreamHandlerOptions* internal_default_instance() {
    return reinterpret_cast<const FixedSizeInputStreamHandlerOptions*>(
               &_FixedSizeInputStreamHandlerOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(FixedSizeInputStreamHandlerOptions& a, FixedSizeInputStreamHandlerOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(FixedSizeInputStreamHandlerOptions* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(FixedSizeInputStreamHandlerOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  FixedSizeInputStreamHandlerOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<FixedSizeInputStreamHandlerOptions>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const FixedSizeInputStreamHandlerOptions& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const FixedSizeInputStreamHandlerOptions& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to, const ::PROTOBUF_NAMESPACE_ID::Message& from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(FixedSizeInputStreamHandlerOptions* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.FixedSizeInputStreamHandlerOptions";
  }
  protected:
  explicit FixedSizeInputStreamHandlerOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kFixedMinSizeFieldNumber = 3,
    kTriggerQueueSizeFieldNumber = 1,
    kTargetQueueSizeFieldNumber = 2,
  };
  // optional bool fixed_min_size = 3 [default = false];
  bool has_fixed_min_size() const;
  private:
  bool _internal_has_fixed_min_size() const;
  public:
  void clear_fixed_min_size();
  bool fixed_min_size() const;
  void set_fixed_min_size(bool value);
  private:
  bool _internal_fixed_min_size() const;
  void _internal_set_fixed_min_size(bool value);
  public:

  // optional int32 trigger_queue_size = 1 [default = 2];
  bool has_trigger_queue_size() const;
  private:
  bool _internal_has_trigger_queue_size() const;
  public:
  void clear_trigger_queue_size();
  int32_t trigger_queue_size() const;
  void set_trigger_queue_size(int32_t value);
  private:
  int32_t _internal_trigger_queue_size() const;
  void _internal_set_trigger_queue_size(int32_t value);
  public:

  // optional int32 target_queue_size = 2 [default = 1];
  bool has_target_queue_size() const;
  private:
  bool _internal_has_target_queue_size() const;
  public:
  void clear_target_queue_size();
  int32_t target_queue_size() const;
  void set_target_queue_size(int32_t value);
  private:
  int32_t _internal_target_queue_size() const;
  void _internal_set_target_queue_size(int32_t value);
  public:

  static const int kExtFieldNumber = 125744319;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::MediaPipeOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FixedSizeInputStreamHandlerOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.FixedSizeInputStreamHandlerOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  bool fixed_min_size_;
  int32_t trigger_queue_size_;
  int32_t target_queue_size_;
  friend struct ::TableStruct_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// FixedSizeInputStreamHandlerOptions

// optional int32 trigger_queue_size = 1 [default = 2];
inline bool FixedSizeInputStreamHandlerOptions::_internal_has_trigger_queue_size() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool FixedSizeInputStreamHandlerOptions::has_trigger_queue_size() const {
  return _internal_has_trigger_queue_size();
}
inline void FixedSizeInputStreamHandlerOptions::clear_trigger_queue_size() {
  trigger_queue_size_ = 2;
  _has_bits_[0] &= ~0x00000002u;
}
inline int32_t FixedSizeInputStreamHandlerOptions::_internal_trigger_queue_size() const {
  return trigger_queue_size_;
}
inline int32_t FixedSizeInputStreamHandlerOptions::trigger_queue_size() const {
  // @@protoc_insertion_point(field_get:mediapipe.FixedSizeInputStreamHandlerOptions.trigger_queue_size)
  return _internal_trigger_queue_size();
}
inline void FixedSizeInputStreamHandlerOptions::_internal_set_trigger_queue_size(int32_t value) {
  _has_bits_[0] |= 0x00000002u;
  trigger_queue_size_ = value;
}
inline void FixedSizeInputStreamHandlerOptions::set_trigger_queue_size(int32_t value) {
  _internal_set_trigger_queue_size(value);
  // @@protoc_insertion_point(field_set:mediapipe.FixedSizeInputStreamHandlerOptions.trigger_queue_size)
}

// optional int32 target_queue_size = 2 [default = 1];
inline bool FixedSizeInputStreamHandlerOptions::_internal_has_target_queue_size() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool FixedSizeInputStreamHandlerOptions::has_target_queue_size() const {
  return _internal_has_target_queue_size();
}
inline void FixedSizeInputStreamHandlerOptions::clear_target_queue_size() {
  target_queue_size_ = 1;
  _has_bits_[0] &= ~0x00000004u;
}
inline int32_t FixedSizeInputStreamHandlerOptions::_internal_target_queue_size() const {
  return target_queue_size_;
}
inline int32_t FixedSizeInputStreamHandlerOptions::target_queue_size() const {
  // @@protoc_insertion_point(field_get:mediapipe.FixedSizeInputStreamHandlerOptions.target_queue_size)
  return _internal_target_queue_size();
}
inline void FixedSizeInputStreamHandlerOptions::_internal_set_target_queue_size(int32_t value) {
  _has_bits_[0] |= 0x00000004u;
  target_queue_size_ = value;
}
inline void FixedSizeInputStreamHandlerOptions::set_target_queue_size(int32_t value) {
  _internal_set_target_queue_size(value);
  // @@protoc_insertion_point(field_set:mediapipe.FixedSizeInputStreamHandlerOptions.target_queue_size)
}

// optional bool fixed_min_size = 3 [default = false];
inline bool FixedSizeInputStreamHandlerOptions::_internal_has_fixed_min_size() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool FixedSizeInputStreamHandlerOptions::has_fixed_min_size() const {
  return _internal_has_fixed_min_size();
}
inline void FixedSizeInputStreamHandlerOptions::clear_fixed_min_size() {
  fixed_min_size_ = false;
  _has_bits_[0] &= ~0x00000001u;
}
inline bool FixedSizeInputStreamHandlerOptions::_internal_fixed_min_size() const {
  return fixed_min_size_;
}
inline bool FixedSizeInputStreamHandlerOptions::fixed_min_size() const {
  // @@protoc_insertion_point(field_get:mediapipe.FixedSizeInputStreamHandlerOptions.fixed_min_size)
  return _internal_fixed_min_size();
}
inline void FixedSizeInputStreamHandlerOptions::_internal_set_fixed_min_size(bool value) {
  _has_bits_[0] |= 0x00000001u;
  fixed_min_size_ = value;
}
inline void FixedSizeInputStreamHandlerOptions::set_fixed_min_size(bool value) {
  _internal_set_fixed_min_size(value);
  // @@protoc_insertion_point(field_set:mediapipe.FixedSizeInputStreamHandlerOptions.fixed_min_size)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fstream_5fhandler_2ffixed_5fsize_5finput_5fstream_5fhandler_2eproto
