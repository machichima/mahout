//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! GPU Buffer Pool for efficient memory reuse
//!
//! Implements a PyTorch-style caching allocator to reduce cudaMalloc/cudaFree overhead.

use std::ffi::c_void;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, DevicePtr};
use crate::error::{MahoutError, Result};

/// Size rounding constants (following PyTorch's strategy)
const SMALL_ALLOCATION: usize = 1 * 1024 * 1024;  // 1 MB
const SMALL_ROUND: usize = 512;                    // 512 bytes
const LARGE_ROUND: usize = 2 * 1024 * 1024;        // 2 MB (GPU huge page)

/// Round allocation size to reduce fragmentation
pub(crate) fn round_size(bytes: usize) -> usize {
    if bytes <= SMALL_ALLOCATION {
        // Small allocation: round to 512 bytes
        ((bytes + SMALL_ROUND - 1) / SMALL_ROUND) * SMALL_ROUND
    } else {
        // Large allocation: round to 2 MB
        ((bytes + LARGE_ROUND - 1) / LARGE_ROUND) * LARGE_ROUND
    }
}

/// Represents a block of GPU memory in the pool
#[cfg(target_os = "linux")]
struct Block {
    ptr: *mut c_void,    // Raw device pointer
    size: usize,         // Size in bytes
    allocated: bool,     // true = in use, false = free
}

/// Statistics for the buffer pool
#[cfg(target_os = "linux")]
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_allocated: usize,   // Total memory allocated from OS
    pub current_in_use: usize,    // Currently borrowed
    pub num_alloc_calls: usize,   // Number of cudaMalloc calls
    pub num_blocks: usize,        // Total blocks in pool
}

/// GPU Buffer Pool for reusing temporary device memory
///
/// Manages a pool of pre-allocated GPU buffers to avoid repeated cudaMalloc/cudaFree overhead.
/// Implements a PyTorch-style caching allocator with size rounding and block splitting.
#[cfg(target_os = "linux")]
pub struct BufferPool {
    /// All blocks ever allocated (both free and in-use)
    blocks: Vec<Block>,

    /// Device for allocating new blocks
    device: Arc<CudaDevice>,

    /// Allocation statistics
    stats: PoolStats,
}

#[cfg(target_os = "linux")]
impl BufferPool {
    /// Create a new buffer pool
    pub fn new(device: Arc<CudaDevice>) -> Self {
        eprintln!("[BufferPool] Initialized");
        Self {
            blocks: Vec::new(),
            device,
            stats: PoolStats {
                total_allocated: 0,
                current_in_use: 0,
                num_alloc_calls: 0,
                num_blocks: 0,
            },
        }
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        let mut stats = self.stats.clone();
        stats.num_blocks = self.blocks.len();
        stats
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self, size_bytes: usize) -> Result<PooledBuffer> {
        let rounded_size = round_size(size_bytes);

        // 1. Try to find a free block that fits (best-fit strategy)
        if let Some(block_idx) = self.find_free_block(rounded_size) {
            self.blocks[block_idx].allocated = true;
            self.stats.current_in_use += self.blocks[block_idx].size;

            return Ok(PooledBuffer {
                ptr: self.blocks[block_idx].ptr,
                size: self.blocks[block_idx].size,
                block_idx,
            });
        }

        // 2. Try to split a larger free block
        if let Some((block_idx, remaining)) = self.try_split_block(rounded_size) {
            return self.use_split_block(block_idx, rounded_size, remaining);
        }

        // 3. No suitable block found - allocate from OS
        self.allocate_new_block(rounded_size)
    }

    /// Release a buffer back to the pool
    pub fn release(&mut self, buffer: PooledBuffer) {
        let block_idx = buffer.block_idx;

        if block_idx >= self.blocks.len() {
            eprintln!("[BufferPool] WARNING: Invalid block index {}", block_idx);
            return;
        }

        let block = &mut self.blocks[block_idx];
        if !block.allocated {
            eprintln!("[BufferPool] WARNING: Double-free detected for block {}", block_idx);
            return;
        }

        block.allocated = false;
        self.stats.current_in_use -= block.size;

        eprintln!("[BufferPool] Released {} bytes ({:.2} MB), in-use: {:.2} MB",
                  block.size, block.size as f64 / 1e6,
                  self.stats.current_in_use as f64 / 1e6);
    }

    fn find_free_block(&self, size: usize) -> Option<usize> {
        // Best-fit: find smallest free block >= size
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.allocated && b.size >= size)
            .min_by_key(|(_, b)| b.size)
            .map(|(idx, _)| idx)
    }

    fn try_split_block(&mut self, size: usize) -> Option<(usize, usize)> {
        // Find a free block larger than needed
        for (idx, block) in self.blocks.iter().enumerate() {
            if !block.allocated && block.size > size + SMALL_ROUND {
                let remaining = block.size - size;
                return Some((idx, remaining));
            }
        }
        None
    }

    fn use_split_block(&mut self, idx: usize, size: usize, remaining: usize)
        -> Result<PooledBuffer> {
        let original_ptr = self.blocks[idx].ptr;
        let original_size = self.blocks[idx].size;

        // First part: allocated
        self.blocks[idx].size = size;
        self.blocks[idx].allocated = true;

        // Second part: free (add as new block)
        let new_ptr = (original_ptr as usize + size) as *mut c_void;
        self.blocks.push(Block {
            ptr: new_ptr,
            size: remaining,
            allocated: false,
        });

        self.stats.current_in_use += size;

        eprintln!("[BufferPool] Split {:.2}MB block into {:.2}MB (used) + {:.2}MB (free)",
                  original_size as f64 / 1e6, size as f64 / 1e6, remaining as f64 / 1e6);

        Ok(PooledBuffer {
            ptr: original_ptr,
            size,
            block_idx: idx,
        })
    }

    fn allocate_new_block(&mut self, size: usize) -> Result<PooledBuffer> {
        eprintln!("[BufferPool] Allocating new block from OS: {} bytes ({:.2} MB)",
                  size, size as f64 / 1e6);

        let buffer = unsafe { self.device.alloc::<u8>(size) }
            .map_err(|e| MahoutError::MemoryAllocation(
                format!("Failed to allocate {} bytes: {:?}", size, e)
            ))?;

        let ptr = *buffer.device_ptr() as *mut c_void;

        let block_idx = self.blocks.len();
        self.blocks.push(Block {
            ptr,
            size,
            allocated: true,
        });

        self.stats.total_allocated += size;
        self.stats.current_in_use += size;
        self.stats.num_alloc_calls += 1;

        // Keep the CudaSlice alive (never drop it until pool is destroyed)
        std::mem::forget(buffer);

        Ok(PooledBuffer {
            ptr,
            size,
            block_idx,
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for BufferPool {
    fn drop(&mut self) {
        eprintln!("[BufferPool] Cleanup:");
        eprintln!("  Total allocated: {:.2} MB",
                  self.stats.total_allocated as f64 / 1e6);
        eprintln!("  Num OS allocations: {}", self.stats.num_alloc_calls);
        eprintln!("  Blocks in pool: {}", self.blocks.len());

        // Free all blocks back to CUDA
        use crate::gpu::cuda_ffi::cuMemFree;
        for (idx, block) in self.blocks.iter().enumerate() {
            unsafe {
                let result = cuMemFree(block.ptr);
                if result != 0 {
                    eprintln!("[BufferPool] WARNING: Failed to free block {}: error {}",
                              idx, result);
                }
            }
        }

        eprintln!("[BufferPool] All memory freed");
    }
}

/// Handle to a borrowed buffer from the pool
#[cfg(target_os = "linux")]
pub struct PooledBuffer {
    ptr: *mut c_void,
    size: usize,
    pub(crate) block_idx: usize,
}

#[cfg(target_os = "linux")]
impl PooledBuffer {
    /// Get device pointer (typed)
    pub fn as_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get raw pointer
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

// Safety: GPU pointers can be sent between threads
#[cfg(target_os = "linux")]
unsafe impl Send for PooledBuffer {}
#[cfg(target_os = "linux")]
unsafe impl Sync for PooledBuffer {}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;

    fn init_test_pool() -> Result<BufferPool> {
        let device = CudaDevice::new(0)
            .map_err(|e| MahoutError::Cuda(format!("Failed to init device: {:?}", e)))?;
        Ok(BufferPool::new(device))
    }

    #[test]
    fn test_round_size() {
        // Small allocations (≤ 1MB) round to 512 bytes
        assert_eq!(round_size(1), 512);
        assert_eq!(round_size(512), 512);
        assert_eq!(round_size(513), 1024);
        assert_eq!(round_size(1024), 1024);

        // Large allocations (> 1MB) round to 2MB
        assert_eq!(round_size(1024 * 1024 + 1), 2 * 1024 * 1024);
        assert_eq!(round_size(2 * 1024 * 1024), 2 * 1024 * 1024);
        assert_eq!(round_size(3 * 1024 * 1024), 4 * 1024 * 1024);
    }

    #[test]
    fn test_pool_basic_allocation() -> Result<()> {
        let mut pool = init_test_pool()?;

        // Allocate a buffer
        let buf1 = pool.acquire(1024)?;
        assert_eq!(buf1.size(), 1024);
        assert!(!buf1.ptr().is_null());

        let stats = pool.stats();
        assert_eq!(stats.num_alloc_calls, 1);
        assert_eq!(stats.current_in_use, 1024);

        // Release it
        pool.release(buf1);

        let stats = pool.stats();
        assert_eq!(stats.current_in_use, 0);

        Ok(())
    }

    #[test]
    fn test_pool_reuse() -> Result<()> {
        let mut pool = init_test_pool()?;

        // First allocation
        let buf1 = pool.acquire(1024)?;
        let ptr1 = buf1.ptr();
        pool.release(buf1);

        // Second allocation should reuse the same block
        let buf2 = pool.acquire(1024)?;
        let ptr2 = buf2.ptr();
        pool.release(buf2);

        assert_eq!(ptr1, ptr2, "Should reuse the same block");

        let stats = pool.stats();
        assert_eq!(stats.num_alloc_calls, 1, "Should only allocate once");

        Ok(())
    }

    #[test]
    fn test_pool_multiple_sizes() -> Result<()> {
        let mut pool = init_test_pool()?;

        let buf1 = pool.acquire(512)?;
        let buf2 = pool.acquire(1024)?;
        let buf3 = pool.acquire(2048)?;

        assert_eq!(buf1.size(), 512);
        assert_eq!(buf2.size(), 1024);
        assert_eq!(buf3.size(), 2048);

        let stats = pool.stats();
        assert_eq!(stats.num_alloc_calls, 3);
        assert_eq!(stats.current_in_use, 512 + 1024 + 2048);

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3);

        let stats = pool.stats();
        assert_eq!(stats.current_in_use, 0);

        Ok(())
    }

    #[test]
    fn test_pool_best_fit() -> Result<()> {
        let mut pool = init_test_pool()?;

        // Allocate and release different sizes
        let buf1 = pool.acquire(512)?;
        let buf2 = pool.acquire(2048)?;
        pool.release(buf1);
        pool.release(buf2);

        // Now pool has free blocks: 512 and 2048
        // Request 1024 should NOT use the 512 block (too small)
        // Should either use 2048 (best fit) or allocate new
        let buf3 = pool.acquire(1024)?;

        let stats = pool.stats();
        // If it split the 2048 block, num_blocks increases
        // If it allocated new, num_alloc_calls increases
        assert!(stats.num_blocks >= 2);

        pool.release(buf3);

        Ok(())
    }

    #[test]
    fn test_pool_concurrent_allocations() -> Result<()> {
        let mut pool = init_test_pool()?;

        // Allocate multiple buffers concurrently (not released)
        let buf1 = pool.acquire(1024)?;
        let buf2 = pool.acquire(1024)?;
        let buf3 = pool.acquire(1024)?;

        let stats = pool.stats();
        assert_eq!(stats.num_alloc_calls, 3, "Should allocate 3 separate blocks");
        assert_eq!(stats.current_in_use, 3 * 1024);

        // Verify they have different pointers
        assert_ne!(buf1.ptr(), buf2.ptr());
        assert_ne!(buf2.ptr(), buf3.ptr());
        assert_ne!(buf1.ptr(), buf3.ptr());

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3);

        Ok(())
    }

    #[test]
    fn test_pool_large_allocation() -> Result<()> {
        let mut pool = init_test_pool()?;

        // Allocate a large buffer (> 1MB)
        let size = 5 * 1024 * 1024; // 5 MB
        let buf = pool.acquire(size)?;

        // Should be rounded up to 6 MB (next multiple of 2MB)
        assert_eq!(buf.size(), 6 * 1024 * 1024);

        let stats = pool.stats();
        assert_eq!(stats.num_alloc_calls, 1);

        pool.release(buf);

        Ok(())
    }

    #[test]
    fn test_pool_stats_accuracy() -> Result<()> {
        let mut pool = init_test_pool()?;

        let buf1 = pool.acquire(1000)?;
        let rounded1 = round_size(1000);

        let stats1 = pool.stats();
        assert_eq!(stats1.total_allocated, rounded1);
        assert_eq!(stats1.current_in_use, rounded1);
        assert_eq!(stats1.num_alloc_calls, 1);

        let buf2 = pool.acquire(2000)?;
        let rounded2 = round_size(2000);

        let stats2 = pool.stats();
        assert_eq!(stats2.total_allocated, rounded1 + rounded2);
        assert_eq!(stats2.current_in_use, rounded1 + rounded2);
        assert_eq!(stats2.num_alloc_calls, 2);

        pool.release(buf1);

        let stats3 = pool.stats();
        assert_eq!(stats3.current_in_use, rounded2);
        assert_eq!(stats3.total_allocated, rounded1 + rounded2); // Total doesn't decrease

        pool.release(buf2);

        let stats4 = pool.stats();
        assert_eq!(stats4.current_in_use, 0);

        Ok(())
    }
}
