#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "FastDeviceMinMax.h"

#include "Logger.h"
#include "CUDAAssert.h"

#include <cstdio>


__device__ float4* BVHTreeNodes;
__device__ float4* TriangleWoopCoordinates;
__device__ int* MappingFromTriangleAddressToIndex;

__device__ void print_float4(float4 printVal) {
	printf("%f, %f, %f, %f\n", printVal.x, printVal.y, printVal.z, printVal.w);
}

#ifdef MAGIC
__device__ __noinline__ void __traceRay(
	Ray rayProperties,
	Hit* rayResultBuffer,
	const float4* startingNode,
	const float4* triNode
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
	printf("triangle node address: 0x%x\n", triNode);
	
	float4 node = *startingNode;
	printf("node content: %f, %f, %f, %f\n", node.x, node.y, node.z, node.w);
	printf("result buffer address: 0x%x\n", rayResultBuffer);
	print_float4((*rayResultBuffer).t_triId_u_v);
	
	printf("anyhit: %d\n", rayProperties.anyhit);

	return;
}
#else

__device__ inline bool RayBoxIntersection(float3 Low, float3 High, float3 InvDir, float3 Ood, float TMin, float TMax, float& OutIntersectionDist)
{
	// ood = RayOrigin * idir;
	
	const float3 lo = Low * InvDir - Ood; // (Low - RayOrigin) / Direction
	const float3 hi = High * InvDir - Ood;
	const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
	
	#ifdef DEBUG
	printf("low: %f, %f, %f\thigh: %f, %f, %f\n", Low.x, Low.y, Low.z, High.x, High.y, High.z);
	printf("lo: %f, %f, %f\thi: %f, %f, %f\n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
	printf("slabMin: %f\tslabMax: %f\n", slabMin, slabMax);
	#endif
	
	OutIntersectionDist = slabMin;

	return slabMin <= slabMax;
}
#endif

__global__ void rtTraceBVH2Plain(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount,
	int* finishedRayCount,
	bool anyhit
)
{


	int rayidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (rayidx >= rayCount)
		return;

	const float4* localBVHTreeNodes = BVHTreeNodes;
	const float4* localTriangleWoopCoordinates = TriangleWoopCoordinates;

#ifdef MAGIC
	// Magic function to traverse BVH tree and test for hits, results stored into result buffer
	// localBVHTreeNodes is set by host in BVHManager.cpp > buildBVH2

	// printf("Starting node %f\n", localBVHTreeNodes);
	// printf("Ray tmin %f\n", rayBuffer[rayidx].origin_tmin.w);
	rayBuffer[rayidx].anyhit = anyhit;

	__traceRay(rayBuffer[rayidx], &rayResultBuffer[rayidx], localBVHTreeNodes, localTriangleWoopCoordinates);
	
	#ifdef DEBUG
	printf("Traced result t: %f\n", rayResultBuffer[rayidx].t_triId_u_v.x);
	#endif

#else

	// Setup traversal + initialisation
	const int EntrypointSentinel = 0x76543210;
	const int STACK_SIZE = 32;
	const float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number

	int traversalStack[STACK_SIZE];
	traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
	int* stackPtr = &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
	int nodeAddr = 0;   // Start from the root.
	int hitAddr = -1;  // No triangle intersected so far.
	int leafAddr = 0;

	float3  idir;    // (1 / ray direction)
	float3	ood;
	float2  triangleuv;

	// Software Aila algorithm

	float3 RayOrigin = make_float3(rayBuffer[rayidx].origin_tmin);
	float3 RayDirection = make_float3(rayBuffer[rayidx].dir_tmax);
	float tmin = rayBuffer[rayidx].origin_tmin.w;
	float hitT = rayBuffer[rayidx].dir_tmax.w;
	
	// ooeps is very small number, used instead of raydir xyz component when that component is near zero
	idir.x = 1.0f / (fabsf(RayDirection.x) > ooeps ? RayDirection.x : copysignf(ooeps, RayDirection.x)); // inverse ray direction
	idir.y = 1.0f / (fabsf(RayDirection.y) > ooeps ? RayDirection.y : copysignf(ooeps, RayDirection.y)); // inverse ray direction
	idir.z = 1.0f / (fabsf(RayDirection.z) > ooeps ? RayDirection.z : copysignf(ooeps, RayDirection.z)); // inverse ray direction
	ood = RayOrigin * idir;
	
	#ifdef DEBUG
	printf("Ray origin: %f, %f, %f\n", RayOrigin.x, RayOrigin.y, RayOrigin.z);
	printf("Ray direction: %f, %f, %f\n", RayDirection.x, RayDirection.y, RayDirection.z);
	printf("Inverse direction: %f, %f, %f\n", idir.x, idir.y, idir.z);
	printf("OOD: %f, %f, %f\n", ood.x, ood.y, ood.z);
	#endif

	// Traversal loop.
	while (nodeAddr != EntrypointSentinel)
	{
		leafAddr = 0;

		while (nodeAddr != EntrypointSentinel && nodeAddr >= 0)
		{
			#ifdef DEBUG
			printf("\n");
			#endif
			
			const float4 n0xy = __ldg(localBVHTreeNodes + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = __ldg(localBVHTreeNodes + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 n01z = __ldg(localBVHTreeNodes + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)

			
			// Are child_index0 and child_index1 next to each other?
			float4 tmp = BVHTreeNodes[nodeAddr + 3]; // child_index0, child_index1
			
			// Convert float4 into 2 float2s?
			int2  cnodes = *(int2*)&tmp;

			const float3 c0lo = make_float3(n0xy.x, n0xy.z, n01z.x);
			const float3 c0hi = make_float3(n0xy.y, n0xy.w, n01z.y);

			const float3 c1lo = make_float3(n1xy.x, n1xy.z, n01z.z);
			const float3 c1hi = make_float3(n1xy.y, n1xy.w, n01z.w);

			float c0dist, c1dist;

			// Ray box test on both child nodes
			bool traverseChild0 = RayBoxIntersection(c0lo, c0hi, idir, ood, tmin, hitT, c0dist);
			bool traverseChild1 = RayBoxIntersection(c1lo, c1hi, idir, ood, tmin, hitT, c1dist);
			
			#ifdef DEBUG
			printf("node addr: 0x%x\n", nodeAddr);
			print_float4(n0xy);
			print_float4(n1xy);
			print_float4(n01z);
			printf("cnodes: 0x%x, 0x%x\n", cnodes.x, cnodes.y);
			printf("C0hit: %d\t C1hit: %d\n", traverseChild0, traverseChild1);
			printf("C0dist: %f\t C1dist: %f\n", c0dist, c1dist);
			#endif
			
			// Check which child is closer?
			bool swp = c1dist < c0dist;

			// If both nodes miss, move to next node in stack
			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *stackPtr;
				stackPtr--;
				// printf("Both children miss; node addr: 0x%x\n", nodeAddr);
			}
			else
			{
				// If first child box hit, use child_index0?
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;
				// printf("Traverse first child; node addr: 0x%x\n", nodeAddr);

				if (traverseChild0 && traverseChild1)
				{
					// If both boxes hit, check which one was closer and swap child_index0 and child_index1?
					if (swp)
						swap(nodeAddr, cnodes.y);

					// Push the farther node to the stack?
					stackPtr++;
					*stackPtr = cnodes.y;
					
					// printf("Both hit, traverse (closer) child; node addr: 0x%x\n", nodeAddr);
				}
			}

			// When is nodeAddr < 0? Are all leaf addresses negative?
			if (nodeAddr < 0 && leafAddr >= 0)
			{
				// Reached leaves. Pop next node from stack?
				leafAddr = nodeAddr;
				nodeAddr = *stackPtr;
				stackPtr--;
				
				// printf("Reached leaves; node addr: %x\n", nodeAddr);
			}

			if (!__any_sync(__activemask(), leafAddr >= 0))
				break;
		}

		#ifdef DEBUG
		printf("Transition to leaves.\n");
		#endif
		
		// Leaf intersections?
		while (leafAddr < 0)
		{
			
			for (int triAddr = ~leafAddr;; triAddr += 3)
			{
				#ifdef DEBUG
				printf("\nLeaf address: 0x%x\t", leafAddr);
				printf("Triangle address: 0x%x\n", triAddr);
				#endif

				// Get vertices?
				float4 v00 = __ldg(localTriangleWoopCoordinates + triAddr + 0);
				float4 v11 = __ldg(localTriangleWoopCoordinates + triAddr + 1);
				float4 v22 = __ldg(localTriangleWoopCoordinates + triAddr + 2);

				// End condition?
				if (__float_as_int(v00.x) == 0x80000000) {
					// printf("%d\n", (*(int*)&v00.x ==  0x80000000));
					break;
				}
					
				#ifdef DEBUG
				// printf("Triangle base: 0x%x, Triangle offset: 0x%x\n", localTriangleWoopCoordinates, triAddr);
				print_float4(v00);
				print_float4(v11);
				print_float4(v22);
				#endif


				// Multi-stage hit algorithm?
				float Oz = v00.w - RayOrigin.x * v00.x - RayOrigin.y * v00.y - RayOrigin.z * v00.z;
				float invDz = 1.0f / (RayDirection.x*v00.x + RayDirection.y*v00.y + RayDirection.z*v00.z);
				float t = Oz * invDz;
				
				#ifdef DEBUG
				printf("t parameter: %f \t", t);
				#endif

				if (t > tmin && t < hitT)
				{
					float Ox = v11.w + RayOrigin.x * v11.x + RayOrigin.y * v11.y + RayOrigin.z * v11.z;
					float Dx = RayDirection.x * v11.x + RayDirection.y * v11.y + RayDirection.z * v11.z;
					float u = Ox + t * Dx;
					
					#ifdef DEBUG
					printf("hit\n");
					printf("u coordinate: %f \t", u);
					#endif

					if (u >= 0.0f && u <= 1.0f)
					{
						float Oy = v22.w + RayOrigin.x * v22.x + RayOrigin.y * v22.y + RayOrigin.z * v22.z;
						float Dy = RayDirection.x * v22.x + RayDirection.y * v22.y + RayDirection.z * v22.z;
						float v = Oy + t*Dy;
						
						#ifdef DEBUG
						printf("hit\n");
						printf("v coordinate: %f \t", v);
						#endif

						if (v >= 0.0f && u + v <= 1.0f)
						{
							triangleuv.x = u;
							triangleuv.y = v;

							hitT = t;
							hitAddr = triAddr;
							
							#ifdef DEBUG
							printf("TRIANGLE HIT: 0x%x at %f\n", triAddr, hitT);
							#endif

							if (anyhit)
							{
								nodeAddr = EntrypointSentinel;
								break;
							}
						}
					}
				}
				
				#ifdef DEBUG
				if (hitT != t) printf("MISS\n");
				#endif
			}

			// Get next node?
			leafAddr = nodeAddr;
			// If leaf node, pop next node from stack?
			if (nodeAddr < 0)
			{
				nodeAddr = *stackPtr;
				stackPtr--;
			}
		}
	}

	// QUESTION: What happens if nothing is hit?
	rayResultBuffer[rayidx].t_triId_u_v = make_float4(
		hitT,
		int_as_float(hitAddr),
		triangleuv.x,
		triangleuv.y
	);
	
	#ifdef DEBUG
		printf("\nResult: \n");
		printf("t: %f, u: %f, v: %f, triangle offset: 0x%x\n", hitT, triangleuv.x, triangleuv.y, hitAddr);
	#endif

#endif

}

__host__ void rtBindBVH2Data(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex)
{
	cudaCheck(cudaMemcpyToSymbol(MappingFromTriangleAddressToIndex, &InMappingFromTriangleAddressToIndex, 1 * sizeof(InMappingFromTriangleAddressToIndex)));
	cudaCheck(cudaMemcpyToSymbol(TriangleWoopCoordinates, &InTriangleWoopCoordinates, 1 * sizeof(InTriangleWoopCoordinates)));
	
	// sizeof(InBVHTreeNodes) == 8, copy address from InBVHNodes to BVHTreeNodes? 
	cudaCheck(cudaMemcpyToSymbol(BVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(InBVHTreeNodes)));
}

__host__ void rtTraceBVH2(
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
	cudaMemset(cudaFinishedRayCount, 0, sizeof(int));

	dim3 blockDim(128, 1);
	dim3 gridDim(idivCeil(rayCount, blockDim.x), 1);

#ifdef ENABLE_PROFILING
	cudaProfilerStart();
	cudaCheck(cudaEventRecord(startEvent, 0));
#endif

   Log("start Aila tracing\n");

	rtTraceBVH2Plain <<< gridDim, blockDim >>> (
		rayBuffer,
		rayResultBuffer,
		rayCount,
		cudaFinishedRayCount,
		anyhit
		);

#ifdef ENABLE_PROFILING
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	Log("%.3fMS, %.2lfMRays/s (rtTraceBVH2 No Dynamic Fetch)", elapsedTime, (double)rayCount / 1000000.0f / (elapsedTime / 1000.0f));

	cudaProfilerStop();
#endif

	cudaFree(cudaFinishedRayCount);
}
