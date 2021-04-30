#pragma once

#include <functional>
#include "EmbreeBVHBuilder.h"

struct GPUBVHIntermediates
{
	std::vector<float4> BVHNodeData;
	std::vector<float4> InlinedPrimitives;
	std::vector<int>    PrimitiveIndices;
};

struct meta
{
	unsigned char upper : 3, lower : 5;
};

struct CWBVHNode
{
	float3 pOrigin;
	char3 e;
	unsigned char imask;
	unsigned int nodeBaseIndex, triBaseIndex;
	uchar3 qlo[8];	
	uchar3 qhi[8];	
	meta childMetaData[8];

};

struct CWBVHIntermediates
{
	std::vector<CWBVHNode> BVHNodeData;
	std::vector<float4> InlinedPrimitives;
	std::vector<int>    PrimitiveIndices;
};

void ConvertToGPUBVH2(
	BVH2Node*& root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	GPUBVHIntermediates& OutIntermediates
);

void ConvertToGPUCompressedWideBVH(
	BVH8Node * root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	GPUBVHIntermediates& OutIntermediates
);

void newConvertToGPUCompressedWideBVH(
	BVH8Node * root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	CWBVHIntermediates& OutIntermediates
);

void WoopifyTriangle(
	float3 v0,
	float3 v1,
	float3 v2,
	float4& OutWoopifiedV0,
	float4& OutWoopifiedV1,
	float4& OutWoopifiedV2
);

