



#define WARP_SIZE 32

__device__ __forceinline__ float shuffleSum32( float sum )
{
    // Reduce final warp using shuffle
    for (unsigned short shift = WARP_SIZE / 2; shift > 0; shift >>= 1)
        sum += __shfl_down( sum, shift );

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum512(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

    if (threadIdx.x < 256)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 256];
    __syncthreads();

    if (threadIdx.x < 128)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 128];
    __syncthreads();

    if (threadIdx.x < 64)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        sum = shuffleSum32( sum );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum256(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

    if (threadIdx.x < 128)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 128];
    __syncthreads();

    if (threadIdx.x < 64)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        sum = shuffleSum32( sum );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum128(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

    if (threadIdx.x < 64)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        sum = shuffleSum32( sum );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

// Parallel sum combining shuffle and shared memory
__device__ __forceinline__ float parallelSum64(
    float* __restrict__ sharedData )
{
    float sum = sharedData[threadIdx.x];

#if (__CUDA_ARCH__ >= 300)
    if (threadIdx.x < 32)
    {
        sum += sharedData[threadIdx.x + 32];
        // Reduce final warp using shuffle
        sum = shuffleSum32( sum );
    }
#else
    // fully unroll reduction within a single warp
    if (threadIdx.x < 32)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 16];
    __syncthreads();

    if (threadIdx.x < 8)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 8];
    __syncthreads();

    if (threadIdx.x < 4)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 4];
    __syncthreads();

    if (threadIdx.x < 2)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 2];
    __syncthreads();

    if (threadIdx.x < 1)
        sharedData[threadIdx.x] = sum = sum + sharedData[threadIdx.x + 1];
    __syncthreads();
#endif

    return sum;
}

