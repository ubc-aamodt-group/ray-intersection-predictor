#include <cuda_profiler_api.h>
#include <stdio.h>

#include "GPUBVHConverter.h"

#include "helper_math.h"
#include "FastDeviceMinMax.h"

#include "Logger.h"
#include "CUDAAssert.h"


__device__ __host__ float uchar4_to_float(unsigned char val1, unsigned char val2, unsigned char val3, unsigned char val4) {
	int float_val = 0;
	
	float_val |= val1;
	float_val |= val2 << 8;
	float_val |= val3 << 16;
	float_val |= val4 << 24;
	
	return *(float *)&float_val;
}

__device__ __host__ float meta_to_float(meta val1, meta val2, meta val3, meta val4) {
	int float_val = 0;
	
	float_val |= val1.lower;
	float_val |= val1.upper << 5;
	float_val |= val2.lower << 8;
	float_val |= val2.upper << 13;
	float_val |= val3.lower << 16;
	float_val |= val3.upper << 21;
	float_val |= val4.lower << 24;
	float_val |= val4.upper << 29;
	
	return *(float *)&float_val;
}

// __device__ unsigned __bfind(unsigned i) { unsigned b; asm volatile("bfind.u32 %0, %1; " : "=r"(b) : "r"(i)); return b; }
__device__ unsigned __bfind(unsigned i) 
{
	unsigned b; 
	asm volatile
	("bfind.u32 %0, %1; " : "=r"(b) : "r"(i)); 
	return b; 
}

__device__ __inline__ uint sign_extend_s8x4(uint i) { uint v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v; }

__device__ __inline__ uint vshl(uint a, uint b, uint c, uint sel) {
	uint v;
	switch (sel) {
	case 0:
		asm("vshl.u32.u32.u32.wrap.add %0, %1.b0, %2.b0, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
		break;
	case 1:
		asm("vshl.u32.u32.u32.wrap.add %0, %1.b1, %2.b1, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
		break;
	case 2:
		asm("vshl.u32.u32.u32.wrap.add %0, %1.b2, %2.b2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
		break;
	case 3:
		asm("vshl.u32.u32.u32.wrap.add %0, %1.b3, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
		break;
	}

	return v;
}

__device__ __inline__ uint extract_byte(uint i, uint n) { return (i >> (n * 8)) & 0xFF; }

__device__ const float4* BVHTreeNodes;
__device__ const float4* TriangleWoopCoordinates;
__device__ const int* MappingFromTriangleAddressToIndex;


#ifdef MAGIC
__device__ __noinline__ void __traceRay(
	Ray rayProperties,
	Hit* rayResultBuffer,
	const CWBVHNode* startingNode,
	const float4* startingTri	
)
{
	printf("Magic Function\n");

/* Ray struct definition
    origin_tmin = make_float4(o_x, o_y, o_z, t_min);
	dir_tmax = make_float4(d_x, d_y, d_z, t_max);
*/
	float3 RayOrigin = make_float3(rayProperties.origin_tmin);
	float3 RayDirection = make_float3(rayProperties.dir_tmax);
	float tmin = rayProperties.origin_tmin.w;
	float tmax = rayProperties.dir_tmax.w;

	printf("tmin: %f\n", tmin);
	printf("tmax: %f\n", tmax);
	printf("ray origin: %f, %f, %f\n", RayOrigin.x, RayOrigin.y, RayOrigin.z);
	printf("ray direction: %f, %f, %f\n", RayDirection.x, RayDirection.y, RayDirection.z);
	printf("starting node address: 0x%x\n", startingNode);
	printf("starting triangle address: 0x%x\n", startingTri);
	printf("result buffer address: 0x%x\n", rayResultBuffer);

	return;
}
#endif

#define DYNAMIC_FETCH 1
#define TRIANGLE_POSTPONING 1

#define STACK_POP(X) { --stackPtr; if (stackPtr < SM_STACK_SIZE) X = traversalStackSM[threadIdx.y][stackPtr][threadIdx.x]; else X = traversalStack[stackPtr - SM_STACK_SIZE]; }
#define STACK_PUSH(X) { if (stackPtr < SM_STACK_SIZE) traversalStackSM[threadIdx.y][stackPtr][threadIdx.x] = X; else traversalStack[stackPtr - SM_STACK_SIZE] = X; stackPtr++; }

__global__ void rtTraceCWBVHDynamicFetch(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount,
	int* finishedRayCount,
	bool anyhit
)
{
	const float ooeps = exp2f(-80.0f);

	const int STACK_SIZE = 32;
	uint2 traversalStack[STACK_SIZE];

	const int SM_STACK_SIZE = 8; // Slightly smaller stack size than the paper (12), as this seems faster on my GTX1080
	__shared__ uint2 traversalStackSM[2][SM_STACK_SIZE][32];

	int rayidx;

	float3 orig, dir;
	float tmin, tmax;
	float idirx, idiry, idirz;
	uint octinv;
	uint2 nodeGroup = make_uint2(0);
	uint2 triangleGroup = make_uint2(0);
	int stackPtr = 0; // char
	int hitAddr = -1;
	float2 triangleuv;
	bool terminated = true;

	__shared__ volatile int nextRayArray[2];

	const float4* localBVHTreeNodes = BVHTreeNodes;
	const float4* localTriangleWoopCoordinates = TriangleWoopCoordinates;

	do
	{
		volatile int& rayBase = nextRayArray[threadIdx.y];

		const unsigned int	maskTerminated = __ballot_sync(__activemask(), terminated);
		const int			numTerminated = __popc(maskTerminated);
		const int			idxTerminated = __popc(maskTerminated & ((1u << threadIdx.x) - 1));
		// printf("terminated: %d, maskTerminated: %d, numTerminated: %d, idxTerminated: %d\n", terminated, maskTerminated, numTerminated, idxTerminated);

    	if (terminated)
		{
			if (idxTerminated == 0)
      		{
				rayBase = atomicAdd(finishedRayCount, numTerminated);
      		}

			rayidx = rayBase + idxTerminated;

			if (rayidx >= rayCount)
			{
				break;
      		}

			orig = make_float3(rayBuffer[rayidx].origin_tmin);
			dir = make_float3(rayBuffer[rayidx].dir_tmax);
			tmin = rayBuffer[rayidx].origin_tmin.w;
			tmax = rayBuffer[rayidx].dir_tmax.w;
			idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x)); // inverse ray direction
			idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y)); // inverse ray direction
			idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z)); // inverse ray direction
			octinv = (dir.x < 0 ? 4 : 0) | (dir.y < 0 ? 2 : 0) | (dir.z < 0 ? 1 : 0);
			octinv = 7 - octinv;
			nodeGroup = make_uint2(0, 0b10000000000000000000000000000000);
			triangleGroup = make_uint2(0);
			stackPtr = 0;
			hitAddr = -1;
			terminated = false;
			
			#ifdef DEBUG
			printf("Ray origin: (%f, %f, %f), Direction: (%f, %f, %f), tmin: %f, tmax: %f\n", orig.x, orig.y, orig.z, dir.x, dir.y, dir.z, tmin, tmax);
			#endif
			
		}
		
	#if DYNAMIC_FETCH
		int lostLoopIterations = 0;
	#endif

		do
		{
			if (nodeGroup.y > 0x00FFFFFF)
			{
				const unsigned int hits = nodeGroup.y;
				const unsigned int imask = nodeGroup.y;
				const unsigned int child_bit_index = __bfind(hits);
				const unsigned int child_node_base_index = nodeGroup.x;

				nodeGroup.y &= ~(1 << child_bit_index);

				if (nodeGroup.y > 0x00FFFFFF)
				{
					STACK_PUSH(nodeGroup);
				}

				{
					const unsigned int slot_index = (child_bit_index - 24) ^ octinv;
					const unsigned int octinv4 = octinv * 0x01010101u;
					const unsigned int relative_index = __popc(imask & ~(0xFFFFFFFF << slot_index));
					const unsigned int child_node_index = child_node_base_index + relative_index;
					#ifdef DEBUG
					printf("child_node_index: 0x%x\n", child_node_index);
					#endif

					float4 n0, n1, n2, n3, n4;
					
					// printf("Child Index: 0x%x\n", child_node_index);
					n0 = __ldg(localBVHTreeNodes + child_node_index * 5 + 0);
					n1 = __ldg(localBVHTreeNodes + child_node_index * 5 + 1);
					n2 = __ldg(localBVHTreeNodes + child_node_index * 5 + 2);
					n3 = __ldg(localBVHTreeNodes + child_node_index * 5 + 3);
					n4 = __ldg(localBVHTreeNodes + child_node_index * 5 + 4);

					float3 p = make_float3(n0);
					int3 e;
					e.x = *((char*)&n0.w + 0);
					e.y = *((char*)&n0.w + 1);
					e.z = *((char*)&n0.w + 2);
					
					#ifdef DEBUG
					printf("P: %f, %f, %f\n", p.x, p.y, p.z);
					printf("e: %i, %i, %i\n", e.x, e.y, e.z);
					#endif

					nodeGroup.x = float_as_uint(n1.x);
					triangleGroup.x = float_as_uint(n1.y);
					triangleGroup.y = 0;
					unsigned int hitmask = 0;
					
					#ifdef DEBUG
					printf("Node Base: 0x%x \tTri Base: 0x%x\n", nodeGroup.x, triangleGroup.x);
					#endif

					const float adjusted_idirx = uint_as_float((e.x + 127) << 23) * idirx;
					const float adjusted_idiry = uint_as_float((e.y + 127) << 23) * idiry;
					const float adjusted_idirz = uint_as_float((e.z + 127) << 23) * idirz;
					const float origx = -(orig.x - p.x) * idirx;
					const float origy = -(orig.y - p.y) * idiry;
					const float origz = -(orig.z - p.z) * idirz;
					#ifdef DEBUG
					// printf("uint_as_float(%d) = %f * %f = %f\n", (e.x + 127)<<23, uint_as_float((e.x + 127) << 23), idirx, adjusted_idirx);
					printf("Ray origin: (%f, %f, %f), Direction: (%f, %f, %f), tmin: %f, tmax: %f\n", orig.x - p.x, orig.y - p.y, orig.z - p.z, adjusted_idirx, adjusted_idiry, adjusted_idirz, tmin, tmax);
					printf("Adjusted origin: (%f, %f, %f)\n", origx, origy, origz);
					
					// printf("n1: %f, %f, %f, %f\n", n1.x, n1.y, n1.z, n1.w);
					// printf("n2: %f, %f, %f, %f\n", n2.x, n2.y, n2.z, n2.w);
					// printf("n3: %f, %f, %f, %f\n", n3.x, n3.y, n3.z, n3.w);
					// printf("n4: %f, %f, %f, %f\n", n4.x, n4.y, n4.z, n4.w);
					#endif
					
					{
						// First 4
						const unsigned int meta4 = float_as_uint(n1.z);
						const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
						const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
						const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
						const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

						// Potential micro-optimization: use PRMT to do the selection here, as described by the paper
						uint swizzledLox = (idirx < 0) ? float_as_uint(n3.z) : float_as_uint(n2.x);
						uint swizzledHix = (idirx < 0) ? float_as_uint(n2.x) : float_as_uint(n3.z);

						uint swizzledLoy = (idiry < 0) ? float_as_uint(n4.x) : float_as_uint(n2.z);
						uint swizzledHiy = (idiry < 0) ? float_as_uint(n2.z) : float_as_uint(n4.x);

						uint swizzledLoz = (idirz < 0) ? float_as_uint(n4.z) : float_as_uint(n3.x);
						uint swizzledHiz = (idirz < 0) ? float_as_uint(n3.x) : float_as_uint(n4.z);

						// printf("Lo: (%d, %d, %d), Hi: (%d, %d, %d)\n", swizzledLox & 0xFF, swizzledLoy & 0xFF, swizzledLoz & 0xFF, swizzledHix & 0xFF, swizzledHiy & 0xFF, swizzledHiz & 0xFF);
						
						float tminx[4];
						float tminy[4];
						float tminz[4];
						float tmaxx[4];
						float tmaxy[4];
						float tmaxz[4];

						tminx[0] = ((swizzledLox >>  0) & 0xFF) * adjusted_idirx + origx;
						tminx[1] = ((swizzledLox >>  8) & 0xFF) * adjusted_idirx + origx;
						tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
						tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

						tminy[0] = ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry + origy;
						// printf("((swizzledLoy >>  0) & 0xFF) * adjusted_idiry(%f) = %f + origy(%f) = %f\n", adjusted_idiry, ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry, origy, tminy[0]);
						tminy[1] = ((swizzledLoy >>  8) & 0xFF) * adjusted_idiry + origy;
						tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
						tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

						tminz[0] = ((swizzledLoz >>  0) & 0xFF) * adjusted_idirz + origz;
						tminz[1] = ((swizzledLoz >>  8) & 0xFF) * adjusted_idirz + origz;
						tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
						tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

						tmaxx[0] = ((swizzledHix >>  0) & 0xFF) * adjusted_idirx + origx;
						tmaxx[1] = ((swizzledHix >>  8) & 0xFF) * adjusted_idirx + origx;
						tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
						tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

						tmaxy[0] = ((swizzledHiy >>  0) & 0xFF) * adjusted_idiry + origy;
						tmaxy[1] = ((swizzledHiy >>  8) & 0xFF) * adjusted_idiry + origy;
						tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
						tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

						tmaxz[0] = ((swizzledHiz >>  0) & 0xFF) * adjusted_idirz + origz;
						tmaxz[1] = ((swizzledHiz >>  8) & 0xFF) * adjusted_idirz + origz;
						tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
						tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

						for (int childIndex = 0; childIndex < 4; childIndex++)
						{
							// Use VMIN, VMAX to compute the slabs
							const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
							const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

							// printf("Bounding box (%d): LO (%d, %d, %d) HI (%d, %d, %d)\n", childIndex, (swizzledLox >> (childIndex*8))&0xff, (swizzledLoy >> (childIndex*8))&0xff, (swizzledLoz >> (childIndex*8))&0xff, (swizzledHix >> (childIndex*8))&0xff, (swizzledHiy >> (childIndex*8))&0xff, (swizzledHiz >> (childIndex*8))&0xff);
							// printf("min: (%f, %f, %f)\tmax:(%f, %f, %f)\n", tminx[childIndex], tminy[childIndex], tminz[childIndex], tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]);
							bool intersected = cmin <= cmax;

							// Potential micro-optimization: use VSHL to implement this part, as described by the paper
							if (intersected)
							{
								// hitmask = vshl(child_bits4, bit_index4, hitmask, childIndex);
								const unsigned int child_bits = extract_byte(child_bits4, childIndex);
								const unsigned int bit_index = extract_byte(bit_index4, childIndex);
								hitmask |= child_bits << bit_index;
								#ifdef DEBUG
								printf("HIT. ChildIndex: %d\n", childIndex);
								#endif
							}
						}
					}

					{
						// Second 4
						const unsigned int meta4 = float_as_uint(n1.w);
						const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
						const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
						const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
						const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;
						
						// printf("meta4:\t0x%x\n", meta4);
						// printf("is_inner4:\t0x%x\n", is_inner4);
						// printf("inner_mask4:\t0x%x\n", inner_mask4);
						// printf("bit_index4:\t0x%x\n", bit_index4);
						// printf("child_bits4:\t0x%x\n", child_bits4);

						// Potential micro-optimization: use PRMT to do the selection here, as described by the paper
						uint swizzledLox = (idirx < 0) ? float_as_uint(n3.w) : float_as_uint(n2.y);
						uint swizzledHix = (idirx < 0) ? float_as_uint(n2.y) : float_as_uint(n3.w);

						uint swizzledLoy = (idiry < 0) ? float_as_uint(n4.y) : float_as_uint(n2.w);
						uint swizzledHiy = (idiry < 0) ? float_as_uint(n2.w) : float_as_uint(n4.y);

						uint swizzledLoz = (idirz < 0) ? float_as_uint(n4.w) : float_as_uint(n3.y);
						uint swizzledHiz = (idirz < 0) ? float_as_uint(n3.y) : float_as_uint(n4.w);
						
						// printf("Lo: (%x, %x, %x), Hi: (%x, %x, %x)\n", swizzledLox, swizzledLoy, swizzledLoz, swizzledHix, swizzledHiy, swizzledHiz);
						// printf("Lo: (%d, %d, %d), Hi: (%d, %d, %d)\n", swizzledLox & 0xFF, swizzledLoy & 0xFF, swizzledLoz & 0xFF, swizzledHix & 0xFF, swizzledHiy & 0xFF, swizzledHiz & 0xFF);

						float tminx[4];
						float tminy[4];
						float tminz[4];
						float tmaxx[4];
						float tmaxy[4];
						float tmaxz[4];

						tminx[0] = ((swizzledLox >>  0) & 0xFF) * adjusted_idirx + origx;
						tminx[1] = ((swizzledLox >>  8) & 0xFF) * adjusted_idirx + origx;
						tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
						tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

						tminy[0] = ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry + origy;
						tminy[1] = ((swizzledLoy >>  8) & 0xFF) * adjusted_idiry + origy;
						tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
						tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

						tminz[0] = ((swizzledLoz >>  0) & 0xFF) * adjusted_idirz + origz;
						tminz[1] = ((swizzledLoz >>  8) & 0xFF) * adjusted_idirz + origz;
						tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
						tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

						tmaxx[0] = ((swizzledHix >>  0) & 0xFF) * adjusted_idirx + origx;
						tmaxx[1] = ((swizzledHix >>  8) & 0xFF) * adjusted_idirx + origx;
						tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
						tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

						tmaxy[0] = ((swizzledHiy >>  0) & 0xFF) * adjusted_idiry + origy;
						tmaxy[1] = ((swizzledHiy >>  8) & 0xFF) * adjusted_idiry + origy;
						tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
						tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

						tmaxz[0] = ((swizzledHiz >>  0) & 0xFF) * adjusted_idirz + origz;
						tmaxz[1] = ((swizzledHiz >>  8) & 0xFF) * adjusted_idirz + origz;
						tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
						tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

						for (int childIndex = 0; childIndex < 4; childIndex++)
						{

							// Use VMIN, VMAX to compute the slabs
							const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
							const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

							// printf("Bounding box (%d): LO (%d, %d, %d) HI (%d, %d, %d)\n", childIndex+4, (swizzledLox >> (childIndex*8))&0xff, (swizzledLoy >> (childIndex*8))&0xff, (swizzledLoz >> (childIndex*8))&0xff, (swizzledHix >> (childIndex*8))&0xff, (swizzledHiy >> (childIndex*8))&0xff, (swizzledHiz >> (childIndex*8))&0xff);
							// printf("min: (%f, %f, %f)\tmax:(%f, %f, %f)\n", tminx[childIndex], tminy[childIndex], tminz[childIndex], tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]);
							bool intersected = cmin <= cmax;

							// Potential micro-optimization: use VSHL to implement this part, as described by the paper
							if (intersected)
							{
								// hitmask = vshl(child_bits4, bit_index4, hitmask, childIndex);
								const unsigned int child_bits = extract_byte(child_bits4, childIndex);
								const unsigned int bit_index = extract_byte(bit_index4, childIndex);
								hitmask |= child_bits << bit_index;
								#ifdef DEBUG
								printf("HIT. ChildIndex: %d\n", childIndex + 4);
								#endif
							}
						}
					} 

					nodeGroup.y = (hitmask & 0xFF000000) | (*((byte*)&n0.w + 3));
					triangleGroup.y = hitmask & 0x00FFFFFF;
					#ifdef DEBUG
					printf("imask: 0x%x\n", (*((byte*)&n0.w + 3)));
					printf("Hit Mask: 0x%x \tnodeGroup.y: 0x%x\n", hitmask, nodeGroup.y);
					#endif
				}
			}
			else
			{
				triangleGroup = nodeGroup;
				nodeGroup = make_uint2(0);
			}
			
		#if TRIANGLE_POSTPONING
			const int totalThreads = __popc(__activemask());
		#endif

			while (triangleGroup.y != 0)
			{
      		#if TRIANGLE_POSTPONING
				const float Rt = 0.2;
				const int threshold = totalThreads * Rt;
				const int numActiveThreads = __popc(__activemask());
				if (numActiveThreads < threshold)
				{
					STACK_PUSH(triangleGroup);
					break;
				}
			#endif

        		int triangleIndex = __bfind(triangleGroup.y);

				int triAddr = triangleGroup.x * 3 + triangleIndex * 3;
				#ifdef DEBUG
				printf("TriAddr: 0x%x\n", triAddr);
				#endif

				float4 v00 = __ldg(localTriangleWoopCoordinates + triAddr + 0);
				float4 v11 = __ldg(localTriangleWoopCoordinates + triAddr + 1);
				float4 v22 = __ldg(localTriangleWoopCoordinates + triAddr + 2);
				
				#ifdef DEBUG
				printf("v00: (%f, %f, %f, %f)\n", v00.x, v00.y, v00.z, v00.w);
				printf("v11: (%f, %f, %f, %f)\n", v11.x, v11.y, v11.z, v11.w);
				printf("v22: (%f, %f, %f, %f)\n", v22.x, v22.y, v22.z, v22.w);
				#endif

				float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
				float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
				float t = Oz * invDz;

				float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
				float Dx = dir.x * v11.x + dir.y * v11.y + dir.z * v11.z;
				float u = Ox + t * Dx;

				float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
				float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
				float v = Oy + t*Dy;

				if (t > tmin && t < tmax)
				{
					if (u >= 0.0f && u <= 1.0f)
					{
						if (v >= 0.0f && u + v <= 1.0f)
						{
							#ifdef DEBUG
							printf("HIT. t: %f\n", t);
							#endif
							triangleuv.x = u;
							triangleuv.y = v;

							tmax = t;
							hitAddr = triAddr;

							if (anyhit) {
								nodeGroup.y = 0;
								triangleGroup.y = 0;
								stackPtr = 0;
								break;
							}
						}
					}
				}

				triangleGroup.y &= ~(1 << triangleIndex);
			}

			if (nodeGroup.y <= 0x00FFFFFF)
			{
				if (stackPtr > 0)
				{
					STACK_POP(nodeGroup);
				}
				else
				{
					if (tmax > 1 && tmax < 5)
					{
					//	printf("test\n");
					}
					rayResultBuffer[rayidx].t_triId_u_v = make_float4(tmax, int_as_float(hitAddr), triangleuv.x, triangleuv.y);
					#ifdef DEBUG
					printf("HIT FOUND. at 0x%x with t %f\n", hitAddr, tmax);
					#endif
					terminated = true;
					break;
				}
			}
		
		#if DYNAMIC_FETCH
			const int Nd = 4;
			const int Nw = 16;
			lostLoopIterations += 32 - __popc(__activemask()) - Nd;
			if (lostLoopIterations >= Nw)
			{
				break;
			}
		#endif
		
		} while (true);

	} while (true);
}


__global__ void newRtTraceCWBVHDynamicFetch(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount,
	int* finishedRayCount,
	bool anyhit
)
{

	const CWBVHNode* localBVHTreeNodes = (CWBVHNode *)BVHTreeNodes;
	const float4* localTriangleWoopCoordinates = TriangleWoopCoordinates;

	int rayidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (rayidx >= rayCount)
		return;


			
			
	#ifdef MAGIC
		__traceRay(rayBuffer[rayidx], &rayResultBuffer[rayidx], localBVHTreeNodes, localTriangleWoopCoordinates);
	
	#else
		
	const float ooeps = exp2f(-80.0f);

	const int STACK_SIZE = 32;
	uint2 traversalStack[STACK_SIZE];

	const int SM_STACK_SIZE = 8; // Slightly smaller stack size than the paper (12), as this seems faster on my GTX1080
	__shared__ uint2 traversalStackSM[2][SM_STACK_SIZE][32];


	__shared__ int nextRayArray[2];
	float3 orig, dir;
	float tmin, tmax;
	float idirx, idiry, idirz;
	uint octinv;
	uint2 nodeGroup = make_uint2(0);
	uint2 triangleGroup = make_uint2(0);
	int stackPtr = 0; // char
	int hitAddr = -1;
	float2 triangleuv;
	
	orig = make_float3(rayBuffer[rayidx].origin_tmin);
	dir = make_float3(rayBuffer[rayidx].dir_tmax);
	tmin = rayBuffer[rayidx].origin_tmin.w;
	tmax = rayBuffer[rayidx].dir_tmax.w;
	idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x)); // inverse ray direction
	idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y)); // inverse ray direction
	idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z)); // inverse ray direction
	octinv = ((dir.x < 0 ? 1 : 0) << 2) | ((dir.y < 0 ? 1 : 0) << 1) | ((dir.z < 0 ? 1 : 0) << 0);
	octinv = 7 - octinv;
	nodeGroup = make_uint2(0, 0b10000000000000000000000000000000);
	triangleGroup = make_uint2(0);
	stackPtr = 0;
	hitAddr = -1;
		
	#if DYNAMIC_FETCH
	int lostLoopIterations = 0;
	#endif

	do
	{
		if (nodeGroup.y > 0x00FFFFFF)
		{
			const unsigned int hits = nodeGroup.y;
			const unsigned int imask = nodeGroup.y;
			const unsigned int child_bit_index = __bfind(hits);
			const unsigned int child_node_base_index = nodeGroup.x;

			nodeGroup.y &= ~(1 << child_bit_index);

			if (nodeGroup.y > 0x00FFFFFF)
			{
				STACK_PUSH(nodeGroup);
			}

			{
				const unsigned int slot_index = (child_bit_index - 24) ^ octinv;
				const unsigned int octinv4 = octinv * 0x01010101u;
				const unsigned int relative_index = __popc(imask & ~(0xFFFFFFFF << slot_index));
				const unsigned int child_node_index = child_node_base_index + relative_index;

				// printf("Child Index: 0x%x\n", child_node_index);
				CWBVHNode current_node = *(CWBVHNode *)(localBVHTreeNodes + child_node_index);

				float3 p = current_node.pOrigin;
				int3 e;
				e.x = current_node.e.x;
				e.y = current_node.e.y;
				e.z = current_node.e.z;
				
				// printf("P: %f, %f, %f\n", p.x, p.y, p.z);
				// printf("e: %i, %i, %i\n", e.x, e.y, e.z);

				nodeGroup.x = current_node.nodeBaseIndex;
				triangleGroup.x = current_node.triBaseIndex;
				triangleGroup.y = 0;
				unsigned int hitmask = 0;
				
				// printf("Node Base: 0x%x \tTri Base: 0x%x\n", nodeGroup.x, triangleGroup.x);

				const float adjusted_idirx = uint_as_float((e.x + 127) << 23) * idirx;
				const float adjusted_idiry = uint_as_float((e.y + 127) << 23) * idiry;
				const float adjusted_idirz = uint_as_float((e.z + 127) << 23) * idirz;
				const float origx = -(orig.x - p.x) * idirx;
				const float origy = -(orig.y - p.y) * idiry;
				const float origz = -(orig.z - p.z) * idirz;

				float4 n1, n2, n3, n4;
				
				n1.x = *(float *)&current_node.nodeBaseIndex;
				n1.y = *(float *)&current_node.triBaseIndex;
				n1.z = meta_to_float(current_node.childMetaData[0], current_node.childMetaData[1], current_node.childMetaData[2], current_node.childMetaData[3]);
				n1.w = meta_to_float(current_node.childMetaData[4], current_node.childMetaData[5], current_node.childMetaData[6], current_node.childMetaData[7]);
				// printf("n1: %f, %f, %f, %f\n", n1.x, n1.y, n1.z, n1.w);
				
				n2.x = uchar4_to_float(current_node.qlo[0].x, current_node.qlo[1].x, current_node.qlo[2].x, current_node.qlo[3].x);
				n2.y = uchar4_to_float(current_node.qlo[4].x, current_node.qlo[5].x, current_node.qlo[6].x, current_node.qlo[7].x);
				n2.z = uchar4_to_float(current_node.qlo[0].y, current_node.qlo[1].y, current_node.qlo[2].y, current_node.qlo[3].y);
				n2.w = uchar4_to_float(current_node.qlo[4].y, current_node.qlo[5].y, current_node.qlo[6].y, current_node.qlo[7].y);
				// printf("n2: %f, %f, %f, %f\n", n2.x, n2.y, n2.z, n2.w);
				
				n3.x = uchar4_to_float(current_node.qlo[0].z, current_node.qlo[1].z, current_node.qlo[2].z, current_node.qlo[3].z);
				n3.y = uchar4_to_float(current_node.qlo[4].z, current_node.qlo[5].z, current_node.qlo[6].z, current_node.qlo[7].z);
				n3.z = uchar4_to_float(current_node.qhi[0].x, current_node.qhi[1].x, current_node.qhi[2].x, current_node.qhi[3].x);
				n3.w = uchar4_to_float(current_node.qhi[4].x, current_node.qhi[5].x, current_node.qhi[6].x, current_node.qhi[7].x);
				// printf("n3: %f, %f, %f, %f\n", n3.x, n3.y, n3.z, n3.w);
				
				n4.x = uchar4_to_float(current_node.qhi[0].y, current_node.qhi[1].y, current_node.qhi[2].y, current_node.qhi[3].y);
				n4.y = uchar4_to_float(current_node.qhi[4].y, current_node.qhi[5].y, current_node.qhi[6].y, current_node.qhi[7].y);
				n4.z = uchar4_to_float(current_node.qhi[0].z, current_node.qhi[1].z, current_node.qhi[2].z, current_node.qhi[3].z);
				n4.w = uchar4_to_float(current_node.qhi[4].z, current_node.qhi[5].z, current_node.qhi[6].z, current_node.qhi[7].z);
				// printf("n4: %f, %f, %f, %f\n", n4.x, n4.y, n4.z, n4.w);
				
				
				{
					// First 4
					const unsigned int meta4 = float_as_uint(n1.z);
					const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
					const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
					const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
					const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

					// Potential micro-optimization: use PRMT to do the selection here, as described by the paper
					uint swizzledLox = (idirx < 0) ? float_as_uint(n3.z) : float_as_uint(n2.x);
					uint swizzledHix = (idirx < 0) ? float_as_uint(n2.x) : float_as_uint(n3.z);

					uint swizzledLoy = (idiry < 0) ? float_as_uint(n4.x) : float_as_uint(n2.z);
					uint swizzledHiy = (idiry < 0) ? float_as_uint(n2.z) : float_as_uint(n4.x);

					uint swizzledLoz = (idirz < 0) ? float_as_uint(n4.z) : float_as_uint(n3.x);
					uint swizzledHiz = (idirz < 0) ? float_as_uint(n3.x) : float_as_uint(n4.z);

					float tminx[4];
					float tminy[4];
					float tminz[4];
					float tmaxx[4];
					float tmaxy[4];
					float tmaxz[4];

					tminx[0] = ((swizzledLox >>  0) & 0xFF) * adjusted_idirx + origx;
					tminx[1] = ((swizzledLox >>  8) & 0xFF) * adjusted_idirx + origx;
					tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
					tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

					tminy[0] = ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry + origy;
					tminy[1] = ((swizzledLoy >>  8) & 0xFF) * adjusted_idiry + origy;
					tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
					tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

					tminz[0] = ((swizzledLoz >>  0) & 0xFF) * adjusted_idirz + origz;
					tminz[1] = ((swizzledLoz >>  8) & 0xFF) * adjusted_idirz + origz;
					tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
					tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

					tmaxx[0] = ((swizzledHix >>  0) & 0xFF) * adjusted_idirx + origx;
					tmaxx[1] = ((swizzledHix >>  8) & 0xFF) * adjusted_idirx + origx;
					tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
					tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

					tmaxy[0] = ((swizzledHiy >>  0) & 0xFF) * adjusted_idiry + origy;
					tmaxy[1] = ((swizzledHiy >>  8) & 0xFF) * adjusted_idiry + origy;
					tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
					tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

					tmaxz[0] = ((swizzledHiz >>  0) & 0xFF) * adjusted_idirz + origz;
					tmaxz[1] = ((swizzledHiz >>  8) & 0xFF) * adjusted_idirz + origz;
					tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
					tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

					for (int childIndex = 0; childIndex < 4; childIndex++)
					{
						// Use VMIN, VMAX to compute the slabs
						const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
						const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

						bool intersected = cmin <= cmax;

						// Potential micro-optimization: use VSHL to implement this part, as described by the paper
						if (intersected)
						{
							const unsigned int child_bits = extract_byte(child_bits4, childIndex);
							const unsigned int bit_index = extract_byte(bit_index4, childIndex);
							hitmask |= child_bits << bit_index;
          				}
					}
				}

				{
					// Second 4
					const unsigned int meta4 = float_as_uint(n1.w);
					const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
					const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
					const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
					const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

					// Potential micro-optimization: use PRMT to do the selection here, as described by the paper
					uint swizzledLox = (idirx < 0) ? float_as_uint(n3.w) : float_as_uint(n2.y);
					uint swizzledHix = (idirx < 0) ? float_as_uint(n2.y) : float_as_uint(n3.w);

					uint swizzledLoy = (idiry < 0) ? float_as_uint(n4.y) : float_as_uint(n2.w);
					uint swizzledHiy = (idiry < 0) ? float_as_uint(n2.w) : float_as_uint(n4.y);

					uint swizzledLoz = (idirz < 0) ? float_as_uint(n4.w) : float_as_uint(n3.y);
					uint swizzledHiz = (idirz < 0) ? float_as_uint(n3.y) : float_as_uint(n4.w);

					float tminx[4];
					float tminy[4];
					float tminz[4];
					float tmaxx[4];
					float tmaxy[4];
					float tmaxz[4];

					tminx[0] = ((swizzledLox >>  0) & 0xFF) * adjusted_idirx + origx;
					tminx[1] = ((swizzledLox >>  8) & 0xFF) * adjusted_idirx + origx;
					tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
					tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

					tminy[0] = ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry + origy;
					tminy[1] = ((swizzledLoy >>  8) & 0xFF) * adjusted_idiry + origy;
					tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
					tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

					tminz[0] = ((swizzledLoz >>  0) & 0xFF) * adjusted_idirz + origz;
					tminz[1] = ((swizzledLoz >>  8) & 0xFF) * adjusted_idirz + origz;
					tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
					tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

					tmaxx[0] = ((swizzledHix >>  0) & 0xFF) * adjusted_idirx + origx;
					tmaxx[1] = ((swizzledHix >>  8) & 0xFF) * adjusted_idirx + origx;
					tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
					tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

					tmaxy[0] = ((swizzledHiy >>  0) & 0xFF) * adjusted_idiry + origy;
					tmaxy[1] = ((swizzledHiy >>  8) & 0xFF) * adjusted_idiry + origy;
					tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
					tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

					tmaxz[0] = ((swizzledHiz >>  0) & 0xFF) * adjusted_idirz + origz;
					tmaxz[1] = ((swizzledHiz >>  8) & 0xFF) * adjusted_idirz + origz;
					tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
					tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

					for (int childIndex = 0; childIndex < 4; childIndex++)
					{

						// Use VMIN, VMAX to compute the slabs
						const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
						const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

						bool intersected = cmin <= cmax;

						// Potential micro-optimization: use VSHL to implement this part, as described by the paper
						if (intersected)
						{
							const unsigned int child_bits = extract_byte(child_bits4, childIndex);
							const unsigned int bit_index = extract_byte(bit_index4, childIndex);
							hitmask |= child_bits << bit_index;
						}
					}
				} 

				// printf("imask: 0x%x\n", current_node.imask);
				nodeGroup.y = (hitmask & 0xFF000000) | current_node.imask;
				// printf("Hit Mask: 0x%x \tnodeGroup.y: 0x%x\n", hitmask, nodeGroup.y);
				triangleGroup.y = hitmask & 0x00FFFFFF;
			}
		}
		else
		{
			triangleGroup = nodeGroup;
			nodeGroup = make_uint2(0);
		}
		
	#if TRIANGLE_POSTPONING
		const int totalThreads = __popc(__activemask());
	#endif

		while (triangleGroup.y != 0)
		{
  		#if TRIANGLE_POSTPONING
			const float Rt = 0.2;
			const int threshold = totalThreads * Rt;
			const int numActiveThreads = __popc(__activemask());
			if (numActiveThreads < threshold)
			{
				STACK_PUSH(triangleGroup);
				break;
			}
		#endif

    		int triangleIndex = __bfind(triangleGroup.y);

			int triAddr = triangleGroup.x * 3 + triangleIndex * 3;

			float4 v00 = __ldg(localTriangleWoopCoordinates + triAddr + 0);
			float4 v11 = __ldg(localTriangleWoopCoordinates + triAddr + 1);
			float4 v22 = __ldg(localTriangleWoopCoordinates + triAddr + 2);

			float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
			float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
			float t = Oz * invDz;

			float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
			float Dx = dir.x * v11.x + dir.y * v11.y + dir.z * v11.z;
			float u = Ox + t * Dx;

			float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
			float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
			float v = Oy + t*Dy;

			if (t > tmin && t < tmax)
			{
				if (u >= 0.0f && u <= 1.0f)
				{
					if (v >= 0.0f && u + v <= 1.0f)
					{
						triangleuv.x = u;
						triangleuv.y = v;

						tmax = t;
						hitAddr = triAddr;

						if (anyhit) {
							nodeGroup.y = 0;
							triangleGroup.y = 0;
							stackPtr = 0;
							break;
						}
					}
				}
			}

			triangleGroup.y &= ~(1 << triangleIndex);
		}

		if (nodeGroup.y <= 0x00FFFFFF)
		{
			if (stackPtr > 0)
			{
				STACK_POP(nodeGroup);
			}
			else
			{
				if (tmax > 1 && tmax < 5)
				{
				//	printf("test\n");
				}
				rayResultBuffer[rayidx].t_triId_u_v = make_float4(tmax, int_as_float(hitAddr), triangleuv.x, triangleuv.y);
				break;
			}
		}
	
	#if DYNAMIC_FETCH
		const int Nd = 4;
		const int Nw = 16;
		lostLoopIterations += 32 - __popc(__activemask()) - Nd;
		if (lostLoopIterations >= Nw)
		{
			break;
		}
	#endif
	
	} while (true);
	#endif
}

__host__ void rtBindCWBVHData(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex)
{
	cudaCheck(cudaMemcpyToSymbol(MappingFromTriangleAddressToIndex, &InMappingFromTriangleAddressToIndex, 1 * sizeof(InMappingFromTriangleAddressToIndex)));
	cudaCheck(cudaMemcpyToSymbol(TriangleWoopCoordinates, &InTriangleWoopCoordinates, 1 * sizeof(InTriangleWoopCoordinates)));
	cudaCheck(cudaMemcpyToSymbol(BVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(InBVHTreeNodes)));
}

__host__ void rtTraceCWBVH(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount,
	bool anyhit
)
{	
#ifdef ENABLE_PROFILING
	float elapsedTime;
	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));
#endif

	int* cudaFinishedRayCount;
	cudaCheck(cudaMalloc(&cudaFinishedRayCount, sizeof(int)));

	dim3 blockDim(32, 2);

#ifdef MAGIC
	printf("dynamic fetch disabled for CWBVH \n");
	dim3 gridDim(idivCeil(rayCount, blockDim.x), 1);
#else
	#if DYNAMIC_FETCH
		dim3 gridDim(32, 32);
	#else
	  printf("dynamic fetch disabled for CWBVH \n");
	  dim3 gridDim(idivCeil(rayCount, blockDim.x), 1);
	#endif
#endif

#ifdef ENABLE_PROFILING
	cudaProfilerStart();
	cudaCheck(cudaEventRecord(startEvent, 0));
#endif
  
	{
		cudaMemset(cudaFinishedRayCount, 0, sizeof(int));
		#ifdef MAGIC
		newRtTraceCWBVHDynamicFetch <<< gridDim, blockDim >>> (
			rayBuffer,
			rayResultBuffer,
			rayCount,
			cudaFinishedRayCount,
			anyhit
			);
		#else
		rtTraceCWBVHDynamicFetch <<< gridDim, blockDim >>> (
			rayBuffer,
			rayResultBuffer,
			rayCount,
			cudaFinishedRayCount,
			anyhit
			);
		#endif
	}

#ifdef ENABLE_PROFILING
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	Log("%.3fMS, %.2fMRays/s (rtTraceCWBVH Dynamic Fetch)", elapsedTime, (float)rayCount / 1000000.0f / (elapsedTime / 1000.0f));

	cudaProfilerStop();
#endif 

	cudaFree(cudaFinishedRayCount);
}
