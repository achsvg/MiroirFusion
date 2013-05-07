#define _GPU_

#include "gpu_def.h"

#define WORK_GROUP_SIZE 64
#define NUM_WORK_GROUP 153600 / WORK_GROUP_SIZE
#define A_SIZE 21
#define B_SIZE 6
 
/// Parallel prefix sum - Compute the sum within 1 work group
/// TODO : avoid bank conflicts

void summand(int px, int idx, int w, 
			__global const float4* cam_est_pose, 	// current estimate
			__global const float2* corresp, 
			__global const float3* vmap,			// in cam space (sensor)
			__global const float3* vmap_prev,		// in global space (raycast)
			__global const float3* nmap_prev,
			__local float* tempA, __local float* tempB)
{
	//int px = 2^thid;//group_idx * 2 + idx;
	
	float2 corresp_p = corresp[px];	// corresponding px in sensor vmap
	float3 v_prev = vmap_prev[px];
	float3 n_prev = nmap_prev[px];
	bool ok = true;
	
	ok = !_isnan2(corresp_p) && !_isnan3(v_prev) && !_isnan3(n_prev);
	
	int corresp_idx = ok?(w * convert_int(corresp_p.y) + convert_int(corresp_p.x)):0;
	float3 corresp_v = vmap[corresp_idx];
	
	ok = ok && !_isnan3(corresp_v) && !isNull(corresp_v);
	
	if(ok)
	{
		float4 est_pose_loc[4] = {cam_est_pose[0], cam_est_pose[1], cam_est_pose[2], cam_est_pose[3]};
		//float4 cam_pose_inv_loc[4] = {cam_pose_inv[0], cam_pose_inv[1], cam_pose_inv[2], cam_pose_inv[3]};
		
		corresp_v = mul_homogenize(est_pose_loc, corresp_v); // to global coordinates
	
		// // compute and add transpose(A) = transpose(G) * nmap_prev(x, y)
		float3 c = cross(n_prev, corresp_v);
		float A[6] = {	
						c.x, //corresp_v.z * n_prev.y - corresp_v.y * n_prev.z, 
						c.y, //-corresp_v.z * n_prev.x + corresp_v.x * n_prev.z,
						c.z, //corresp_v.y * n_prev.x - corresp_v.x * n_prev.y,
						n_prev.x,
						n_prev.y,
						n_prev.z
					};
		
		int k = 0;
		for(int i = 0; i < 6; i++)			// cols
		{
			for(int j = i; j < 6; j++)		// rows
				tempA[idx*A_SIZE + (k++)] = A[i] * A[j];		// 21 elements of the upper triangular 6x6 matrix
		}
		
		// compute b = transpose(nmap_prev(x,y)) * (vmap_prev(x,y) - vmap(x,y))
		// add sumB += transpose(A) * b
		float b = dot(n_prev, v_prev - corresp_v);
		
		for(int i = 0; i < 6; i++)
			tempB[idx*B_SIZE + i] = A[i] * b;		// B is 6x1
	}
	else
	{
		for(int k = 0; k < A_SIZE; k++)			
			tempA[idx*A_SIZE + k] = 0.0f;
		for(int i = 0; i < B_SIZE; i++)
			tempB[idx*B_SIZE + i] = 0.0f;		
	}
}

void reduce
(
	int local_id,
	int thid,
	__local float* tempA, 
	__local float* tempB, 
	__global float* sumA, 
	__global float* sumB, 
	int local_size,
	int n
)
{
	int offset = 1;
	
	for(int d = n >> 1; d > 0; d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(local_id < d)
		{
			int ai = offset * (2 * local_id + 1) - 1;
			int bi = offset * (2 * local_id + 2) - 1;
			
			for(int i = 0; i < A_SIZE; i++)
				tempA[bi*A_SIZE+i] += tempA[ai*A_SIZE+i];
			for(int i = 0; i < B_SIZE; i++)
				tempB[bi*B_SIZE+i] += tempB[ai*B_SIZE+i];
		}
		offset *= 2;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_id == 0)
	{
		int group = thid/local_size;
		
		// the last elements in tempA and tempB contain the sum of the work-group
		for(int i = 0; i < A_SIZE; i++)
			sumA[group*A_SIZE+i] = tempA[(n-1)*A_SIZE+i];
		for(int i = 0; i < B_SIZE; i++)
			sumB[group*B_SIZE+i] = tempB[(n-1)*B_SIZE+i];
	}
}

__kernel
void compute_sums
(
	__global int* size,
	__global float* ABlocks,	// size*A_SIZE 
	__global float* BBlocks,	// size*B_SIZE 
	__global float* A,			// (size / 2 + pad) / WORK_GROUP_SIZE
	__global float* B
)
{
	__local float tempA[2*WORK_GROUP_SIZE*A_SIZE];
	__local float tempB[2*WORK_GROUP_SIZE*B_SIZE];
	
	const int local_id = get_local_id(0);
	const int thid = get_global_id(0);
	//int group_idx = 2 * thid / WORK_GROUP_SIZE * WORK_GROUP_SIZE;
	
	if(thid < ((*size) / 2)) // max thid = 1215, size = 2400, size/2 = 1200
	{
		for(int i = 0; i < A_SIZE; i++)
		{
			tempA[2*local_id*A_SIZE+i] = ABlocks[(thid*2)*A_SIZE+i];
			tempA[(2*local_id+1)*A_SIZE+i] = ABlocks[(thid*2+1)*A_SIZE+i];
		}
		
		for(int i = 0; i < B_SIZE; i++)
		{
			tempB[2*local_id*B_SIZE+i] = BBlocks[(thid*2)*B_SIZE+i];
			tempB[(2*local_id+1)*B_SIZE+i] = BBlocks[(thid*2+1)*B_SIZE+i];
		}
	}
	else 
	{
		for(int i = 0; i < A_SIZE; i++)
		{
			tempA[2*local_id*A_SIZE+i] = 0.0f;
			tempA[(2*local_id+1)*A_SIZE+i] = 0.0f;
		}
		
		for(int i = 0; i < B_SIZE; i++)
		{
			tempB[2*local_id*B_SIZE+i] = 0.0f;
			tempB[(2*local_id+1)*B_SIZE+i] = 0.0f;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	reduce(local_id, thid, tempA, tempB, A, B, WORK_GROUP_SIZE, WORK_GROUP_SIZE*2);
	
	
}

__kernel 
void compute_block_sums
(
	__global const int* width,
	__global const float4* cam_est_pose,
	__global const float2* corresp,
	__global const float3* vmap,		// in cam space
	//__global const float3* nmap,
	__global const float3* vmap_prev,	// in global space
	__global const float3* nmap_prev,
	__global float* sumA,				// array of size NUM_WORK_GROUP*A_SIZE
	__global float* sumB				// array of size NUM_WORK_GROUP*B_SIZE
)
{
	__local float tempA[2*WORK_GROUP_SIZE*A_SIZE];	// hold the prefix sum for the current work group		
	__local float tempB[2*WORK_GROUP_SIZE*B_SIZE]; 
	const int local_id = get_local_id(0);
	const int thid = get_global_id(0);
	
	int w = *width;
	//int group_idx = thid / WORK_GROUP_SIZE * WORK_GROUP_SIZE;// - 1;//thid - (thid%WORK_GROUP_SIZE);
	
	summand(thid*2, 2*local_id, w, cam_est_pose, corresp, vmap, vmap_prev, nmap_prev, tempA, tempB);
	summand(thid*2+1, 2*local_id+1, w, cam_est_pose, corresp, vmap, vmap_prev, nmap_prev, tempA, tempB);
	
	reduce(local_id, thid, tempA, tempB, sumA, sumB, WORK_GROUP_SIZE, WORK_GROUP_SIZE*2);
}