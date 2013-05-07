#define _GPU_

#include "gpu_def.h"

__kernel void TSDF(	__global short2* F,
					__global const int* width, 
					__global const int* height,
					//__global float4 cam_pos[4],
					__global float4* cam_pos_inv,
					__global ushort* dmap,
					__global ulong* k,				// current frame
					__global const float* fl,		// focal length
					__constant struct TsdfParams* params
				)
{
	// go through all the depth value and update their associated voxel TSDF
	const int voxel_x = get_global_id(0);
	const int voxel_y = get_global_id(1);
	float focal_length = *fl;
	//ulong frame = *k;
	
	float4 cam_pos_inv_l[4] = { cam_pos_inv[0], cam_pos_inv[1], cam_pos_inv[2], cam_pos_inv[3] };
	int w = *width;
	int h = *height;
	
	struct TsdfParams l_params = *params;
	
	int3 resolution = {convert_int(l_params.resolution[0]), convert_int(l_params.resolution[1]), convert_int(l_params.resolution[2])};
	float mu = l_params.mu;
	
	float3 cellsize = { l_params.volume[0] / l_params.resolution[0], l_params.volume[1] / l_params.resolution[1], l_params.volume[2] / l_params.resolution[2] };
	
	int xyIdx = voxel_x*resolution.z + voxel_y*resolution.x*resolution.z;
	
	// OPTIMIZATION TRICK: get first voxel to precompute some of the matrix-vector product since all these voxels share the same x & y coordinates
	float4 voxel0 = getVoxelGlobal(convert_float(voxel_x), convert_float(voxel_y), 0.0f, cellsize);
	voxel0.x -= l_params.volume[0] / 2.0f;
	voxel0.y -= l_params.volume[1] / 2.0f;
	voxel0.w = 0.0f;
	//float4 baseVoxel = mul(cam_pos_inv_l, voxel0);	// current cam space
	float4 baseVoxel = voxel0 + cam_pos_inv_l[3];
	baseVoxel.z = 0.0f;	// will contribute later in the loop
	baseVoxel.w = 0.0f;
	baseVoxel = mul(cam_pos_inv_l, baseVoxel);
	
	for(int voxel_z = 0; voxel_z < resolution.z; voxel_z++)
	{
		float voxel_z_g = (convert_float(voxel_z) + 0.5f) * cellsize.z;
		
		//float4 voxel_c = mul(cam_pos_inv_l,voxel_g);	// voxel in camera space, bottleneck
		float4 voxel_c = { 	baseVoxel.x + cam_pos_inv_l[2].x * (voxel_z_g + cam_pos_inv_l[3].z),
							baseVoxel.y + cam_pos_inv_l[2].y * (voxel_z_g + cam_pos_inv_l[3].z),
							baseVoxel.z + cam_pos_inv_l[2].z * (voxel_z_g + cam_pos_inv_l[3].z),
							baseVoxel.w + cam_pos_inv_l[2].w * (voxel_z_g + cam_pos_inv_l[3].z)
						 };	// optimization trick
		
		// if(voxel_c.z <= 0.0f)	// behind camera
			// break;
		
		int2 pix = { convert_int((voxel_c.x / voxel_c.z) * focal_length + convert_float(w)/2.0f), convert_int((voxel_c.y / voxel_c.z) * focal_length + convert_float(h)/2.0f)}; 
		int idx = voxel_z + xyIdx;
		
		if( pix.x >= 0 && pix.y >= 0 && pix.x < w && pix.y < h && idx >= 0 && idx < resolution.x * resolution.y * resolution.z )
		{
			ushort depth = dmap[pix.y * w + pix.x];	
			
			if(depth != 0)
			{
				float x = (convert_float(pix.x) - convert_float(w)/2.0f)/focal_length;
				float y = (convert_float(pix.y) - convert_float(h)/2.0f)/focal_length;
				float lambda_inv = sqrt( x*x + y*y + 1.0f );
			
				float3 voxel3_c = {voxel_c.x, voxel_c.y, voxel_c.z};
				
				float dist = sqrt(voxel3_c.x * voxel3_c.x + voxel3_c.y * voxel3_c.y + voxel3_c.z * voxel3_c.z); // distance(t, voxel3_g) is slow?
				float arg = convert_float(depth) - dist / lambda_inv;
				
				if(arg >= -mu)
				{
					float F_Rk = arg/mu; 
					F_Rk = max( -1.0f, min( 1.0f, F_Rk ) );
					
					float2 cur_val = unpack_tsdf(F[idx]);	// bottleneck
					float weight = cur_val.y;
					if(weight < convert_float(SHORT_MAX * 2))	// weight limit 
					{
						float val = ((weight * cur_val.x) + F_Rk) / (weight+1.0f);
						pack_tsdf(max( -1.0f, min( 1.0f, val ) ), weight+1.0f, F, idx); // bottleneck
					}
				}
			}
		}
	}
}