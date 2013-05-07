#define _GPU_

#include "gpu_def.h"

// Projective point-plane data association
// Finds corresponding vertices between raycasted volume and current measurement
// given an estimated camera pose (initialised with previous camera pose).

// __kernel void find_correspondences
// (
	// __global const int* width,
	// __global const int* height,
	// __global const float* fl,
	// __global const float4* cam_pose,
	// __global const float4* cam_est_pose,			// estimated transform for the new frame
	// __global const float4* cam_inc_trans,		// transform from current pose to estimated pose
	// __global const float3* vmap_raycast,		// in global space
	// __global const float3* vmap_sensor,			// in camera space
	// __global const float3* nmap_raycast,
	// __global const float3* nmap_sensor,
	// __global float2* vcorresp,
	// __global uint* counter
// )
// {
	// // for each image pixel (x,y)
	// const int x = get_global_id(0);
	// const int y = get_global_id(1);
	
	// int w = *width;
	// int h = *height;
	// float focal_length = *fl;
	
	// int idx = w * y + x;
	// //float3 vprev_tmp = vmap_raycast[idx];
	// //float4 vprev = { vprev_tmp.x, vprev_tmp.y, vprev_tmp.z, 1 };
	// float3 v_tmp = vmap_sensor[idx];
	// float4 v = { v_tmp.x, v_tmp.y, v_tmp.z, 1 };	
		
	// if(!_isnan(v))
	// {
		
		// float4 inc_trans_l[4] = { cam_inc_trans[0], cam_inc_trans[1], cam_inc_trans[2], cam_inc_trans[3] } ;
		// float4 p = mul(inc_trans_l, v);
		
		// p.x = focal_length * p.x / p.z + convert_float(w)/2.0f;
		// p.y = focal_length * p.y / p.z + convert_float(h)/2.0f;
		
		// int px = round(p.x);
		// int py = round(p.y);
		
		// if(px >= 0.0f && py >= 0.0f && px < w && py < h)
		// {
			// float4 est_pose_l[4] = { cam_est_pose[0], cam_est_pose[1], cam_est_pose[2], cam_est_pose[3] };
			// float4 cam_pose_l[4] = { cam_pose[0], cam_pose[1], cam_pose[2], cam_pose[3] };
			// float3 v_rc_tmp = vmap_raycast[w * py + px];
			
			// if(!_isnan(v_rc_tmp))
			// {
				// float4 v_rc = {v_rc_tmp.x, v_rc_tmp.y, v_rc_tmp.z, 1.0f};
				
				// float4 diff = mul(est_pose_l, v) - mul(cam_pose_l, v_rc); 
				// float dist = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
				
				// float3 norm_s = nmap_sensor[idx];
				// float3 rot_norm = 
				// {
					// est_pose_l[0].x * norm_s.x + est_pose_l[1].x * norm_s.y + est_pose_l[2].x * norm_s.z,
					// est_pose_l[0].y * norm_s.x + est_pose_l[1].y * norm_s.y + est_pose_l[2].y * norm_s.z,
					// est_pose_l[0].z * norm_s.x + est_pose_l[1].z * norm_s.y + est_pose_l[2].z * norm_s.z
				// }; 
				
				// float3 norm_rc = nmap_raycast[w * py + px];
				// float4 norm_rc_tmp = {norm_rc.x, norm_rc.y, norm_rc.z, 0.0f};
				// norm_rc_tmp = mul(cam_pose_l, norm_rc_tmp);
				// norm_rc.x = norm_rc_tmp.x;
				// norm_rc.y = norm_rc_tmp.y;
				// norm_rc.z = norm_rc_tmp.z;
				
				// if(dist < 10.0f && fabs((float)dot(normalize(rot_norm), norm_rc)) > 0.5f)	// in mm
				// {
					// vcorresp[idx].x = px;
					// vcorresp[idx].y = py;
					// atom_inc(counter);
					// return;
				// }
			// }
		// }
	// }
	// vcorresp[idx].x = NAN;	
	// vcorresp[idx].y = NAN;
// }

#define ANGLE_THRES 0.006092310f	// sin(20*pi/180)
#define DIST_THRES 10.0f

__kernel void find_correspondences
(
	__global const int* width,
	__global const int* height,
	__global const float* fl,
	__global const float4* cam_pose_inv_p,
	__global const float3* vmap_raycast,		// in global space
	__global const float3* vmap_sensor,			// in camera space
	__global const float3* nmap_raycast,
	__global const float3* nmap_sensor,
	__global float2* vcorresp,
	__global uint* counter
)
{
	// for each image pixel (x,y)
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	int w = *width;
	int h = *height;
	float focal_length = *fl;
	
	int idx = w * y + x;
	//float3 vprev_tmp = vmap_raycast[idx];
	//float4 vprev = { vprev_tmp.x, vprev_tmp.y, vprev_tmp.z, 1 };
	float3 v_tmp = vmap_raycast[idx];
	float4 v_prev = { v_tmp.x, v_tmp.y, v_tmp.z, 0.0f };	
		
	if(!_isnan4(v_prev))
	{
		float4 cam_pose_inv[4] = {cam_pose_inv_p[0], cam_pose_inv_p[1], cam_pose_inv_p[2], cam_pose_inv_p[3]};
		
		float4 v_prev_cs = v_prev + cam_pose_inv[3];
		v_prev_cs.w = 0.0f;
		v_prev_cs = mul(cam_pose_inv, v_prev_cs);	// to cam space
		
		float2 p = { focal_length * v_prev_cs.x / v_prev_cs.z + convert_float(w)/2.0f,		// perspective project
					 focal_length * v_prev_cs.y / v_prev_cs.z + convert_float(h)/2.0f };
		
		int px = round(p.x);
		int py = round(p.y);
		
		if(px >= 0.0f && py >= 0.0f && px < w && py < h)
		{
			float3 v_corresp_tmp = vmap_sensor[w * py + px];	
			
			if(!_isnan3(v_corresp_tmp))
			{
				float4 v_corresp_cs = {v_corresp_tmp.x, v_corresp_tmp.y, v_corresp_tmp.z, 1.0f}; // in cam space
				
				float4 diff = v_corresp_cs - v_prev_cs; 
				float dist = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
				
				float3 norm_rc = nmap_raycast[idx];
				float4 norm_rc_4 = {norm_rc.x, norm_rc.y, norm_rc.z, 0.0f};
				float4 rot_norm_4 = mul(cam_pose_inv, norm_rc_4);	// to cam space
				float3 rot_norm = {rot_norm_4.x, rot_norm_4.y, rot_norm_4.z};
				float3 norm_s = nmap_sensor[w * py + px];
				
				
				if(dist < DIST_THRES && fabs((float)dot(normalize(rot_norm), norm_s)) > ANGLE_THRES)	// in mm
				{
					vcorresp[idx].x = px;
					vcorresp[idx].y = py;
					//atom_inc(counter);
					return;
				}
			}
		}
	}
	vcorresp[idx].x = NAN;	
	vcorresp[idx].y = NAN;
}