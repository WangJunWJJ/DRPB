// Modified by wangjun
//not finished
#ifndef DistributedData_CC_CHUNK_STORE_H_
#define REVERB_CC_CHUNK_STORE_H_

#include <memory>
#include <utility>
#include <vector>
#include <cstdint>
#include "proto/schema.pb.h"
#include "absl/base/internal/invoke.h"
#include "absl/base/internal/low_level_scheduling.h"
#include "absl/base/internal/raw_logging.h"
#include "absl/base/internal/scheduling_mode.h"
#include "absl/base/internal/spinlock_wait.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/base/port.h"


namespace DRPB {

class ChunkStore {
 public:
  using Key = uint64_t;

  class Chunk {
   public:
    explicit Chunk(ChunkData data);

    // Unique identifier of the chunk.
    uint64_t key() const;

    // Returns the proto data of the chunk.
    const ChunkData& data() const;

    // (Potentially cached) size of `data`.
    size_t DataByteSizeLong() const;

    // Alias for `data().sequence_range().episode_id()`.
    uint64_t episode_id() const;

    // The number of tensors batched together in each column. Note that all
    // columns always share the same number of rows (i.e batch dimension).
    int32_t num_rows() const;

    // Number of tensors in each step.
    int num_columns() const;

   private:
    ChunkData data_;
    mutable size_t data_byte_size_;
    mutable absl::once_flag data_byte_size_once_;
  };

  // Starts `cleaner_`. `cleanup_batch_size` is the number of keys the cleaner
  // should wait for before acquiring the lock and erasing them from `data_`.
  explicit ChunkStore(int cleanup_batch_size = 1000);

  // Stops `cleaner_` closes `delete_keys_`.
  ~ChunkStore();

  // Attempts to insert a Chunk into the map using the key inside `item`. If no
  // entry existed for the key, a new Chunk is created, inserted and returned.
  // Otherwise, the existing chunk is returned.
  std::shared_ptr<Chunk> Insert(ChunkData item) ABSL_LOCKS_EXCLUDED(mu_);

  // Gets the Chunk for each given key. Returns an error if one of the items
  // does not exist or if `Close` has been called. On success, the returned
  // items are in the same order as given in `keys`.
  absl::status Get(absl::Span<const Key> keys,
                         std::vector<std::shared_ptr<Chunk>>* chunks)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Blocks until `num_chunks` expired entries have been cleaned up from
  // `data_`. This method is called automatically by a background thread to
  // limit memory size, but does not have any effect on the semantics of Get()
  // or Insert() calls.
  //
  // Returns false if `delete_keys_` closed before `num_chunks` could be popped.
  bool CleanupInternal(int num_chunks) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  // Gets an item. Returns nullptr if the item does not exist.
  std::shared_ptr<Chunk> GetItem(Key key) ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Holds the actual mapping of key to Chunk. We only hold a weak pointer to
  // the Chunk, which means that destruction and reference counting of the
  // chunks happens independently of this map.
  internal::flat_hash_map<Key, std::weak_ptr<Chunk>> data_ ABSL_GUARDED_BY(mu_);

  // Mutex protecting access to `data_`.
  mutable absl::Mutex mu_;

  // Queue of keys of deleted items that will be cleaned up by `cleaner_`. Note
  // the queue have to be allocated on the heap in order to avoid dereferencing
  // errors caused by a stack allocated ChunkStore getting destroyed before all
  // Chunk have been destroyed.
  std::shared_ptr<internal::Queue<Key>> delete_keys_;

  // Consumes `delete_keys_` to remove dead pointers in `data_`.
  std::unique_ptr<internal::Thread> cleaner_;
};

}  // namespace drpb

#endif  // drpb_CC_CHUNK_STORE_H_
