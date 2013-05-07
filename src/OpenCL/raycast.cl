#define _GPU_

#include "gpu_def.h"

int3 getVoxel(float4 p, struct TsdfParams params)
{
	int3 res = { convert_int(p.x*params.resolution[0]/params.volume[0] + params.resolution[0]/2.0f), 
				 convert_int(p.y*params.resolution[1]/params.volume[1] + params.resolution[1]/2.0f), 
				 convert_int(p.z*params.resolution[2]/params.volume[2]) };	
	// int3 res = { ((p.x)*params.resolution[0]/params.volume[0] + params.resolution[0]/2.0f), 
				 // ((p.y)*params.resolution[1]/params.volume[1] + params.resolution[1]/2.0f), 
				 // (p.z*params.resolution[2]/params.volume[2]) };	
	return res;
}

float3 reflect(float3 V, float3 N){
        return V - 2.0f * dot( V, N ) * N;
}

float getTSDF(
	__global short2* F,
	int3 resolution,
	int3 voxel,
	int x_offset,
	int y_offset,
	int z_offset
)
{
	int ox = (voxel.x + x_offset);
	int oy = (voxel.y + y_offset);
	int oz = (voxel.z + z_offset);
	
	
	//int3 resolution = { (l_params.resolution[0]), (l_params.resolution[1]), (l_params.resolution[2]) };
	
	int idx = ox*resolution.z + oy*resolution.x*resolution.z + oz;
	if( ox < 0 || ox >= resolution.x ||
		oy < 0 || oy >= resolution.y ||
		oz < 0 || oz >= resolution.z )
		return NAN;
		
	return unpack_tsdf(F[idx]).x;
}

float interpolateTrilinearly(
	__global short2* F,
	float4 point,
	int3 voxel,
	float3 cellsize,
	struct TsdfParams params
)
{
	int3 resolution = { convert_int(params.resolution[0]), convert_int(params.resolution[1]), convert_int(params.resolution[2]) };
	
	if (voxel.x <= 0 || voxel.x >= resolution.x - 1)
	  return NAN;

	if (voxel.y <= 0 || voxel.y >= resolution.y - 1)
	  return NAN;

	if (voxel.z <= 0 || voxel.z >= resolution.z - 1)
	  return NAN;
	  
	float vx = ((float)voxel.x + 0.5f) * cellsize.x - params.volume[0]/2.0f;
	float vy = ((float)voxel.y + 0.5f) * cellsize.y - params.volume[1]/2.0f;
	float vz = ((float)voxel.z + 0.5f) * cellsize.z;

	voxel.x = (point.x < vx) ? (voxel.x - 1) : voxel.x;
	voxel.y = (point.y < vy) ? (voxel.y - 1) : voxel.y;
	voxel.z = (point.z < vz) ? (voxel.z - 1) : voxel.z;

	float a = (point.x - ((convert_float(voxel.x) + 0.5f) * cellsize.x - params.volume[0]/2.0f)) / cellsize.x;
	float b = (point.y - ((convert_float(voxel.y) + 0.5f) * cellsize.y - params.volume[1]/2.0f)) / cellsize.y;
	float c = (point.z - (convert_float(voxel.z) + 0.5f) * cellsize.z) / cellsize.z;
	
	float V000 = getTSDF(F, resolution, voxel, 0, 0, 0);
	float V001 = getTSDF(F, resolution, voxel, 0, 0, 1);
	float V010 = getTSDF(F, resolution, voxel, 0, 1, 0);
	float V011 = getTSDF(F, resolution, voxel, 0, 1, 1);
	float V100 = getTSDF(F, resolution, voxel, 1, 0, 0);
	float V101 = getTSDF(F, resolution, voxel, 1, 0, 1);
	float V110 = getTSDF(F, resolution, voxel, 1, 1, 0);
	float V111 = getTSDF(F, resolution, voxel, 1, 1, 1);
	
	if(isnan(V000) || isnan(V001) || isnan(V010) || isnan(V011) || isnan(V100) || isnan(V101) || isnan(V110) || isnan(V111))
		return NAN;
	
	float res = V000 * (1.0f - a) * (1.0f - b) * (1.0f - c) +
				V001 * (1.0f - a) * (1.0f - b) * c +
				V010 * (1.0f - a) * b * (1.0f - c) +
				V011 * (1.0f - a) * b * c +
				V100 * a * (1.0f - b) * (1.0f - c) +
				V101 * a * (1.0f- b) * c +
				V110 * a * b * (1.0f - c) +
				V111 * a * b * c;
	return res;
}

__kernel void raycast(
	__global short2* F,
	__global const int* width,
	__global const int* height,
	__global const float4* cam_pos,
	__global const float* fl,		// focal length
	__global const float2* ppt,		// principal point 
	__global float3* vmap,
	__global float3* nmap,
#ifdef MIROIR_GL_INTEROP
	__write_only image2d_t frameBuffer,
#endif
	__constant struct TsdfParams* params
)
{
	// cast a ray for each pixel (u,v)
	const int u = get_global_id(0);
	const int v = get_global_id(1);
	
	int w = *width;
	int h = *height;
	
	float4 cam_pos_l[4] = { cam_pos[0], cam_pos[1], cam_pos[2], cam_pos[3] };
	
	float depth_march = 0.0f;
	float focal_length = *fl;
	
	// TODO : start ray where the actual volume starts! (ie intersect ray with volume)
	float4 ray_start = {convert_float(u-w/2.0f)/focal_length, convert_float(v-h/2.0f)/focal_length,1.0f,1.0f};	// in camera space
	ray_start = mul(cam_pos_l, ray_start);	// in global space
	float4 vertex_4 = ray_start;
			
	float4 ray_dir = normalize(vertex_4 - cam_pos_l[3]);
	
	//ensure that it isn't a degenerate case
	// ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
	// ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
	// ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;
	
	struct TsdfParams l_params = *params;
	
	int pix_idx = w*v+u;
	
	// kinect sensor range is ~[0.4,8]
	float step = l_params.mu*0.8f;
		
	float4 ray_step = ray_dir*step;
	
	float3 cellsize = { l_params.volume[0] / l_params.resolution[0], l_params.volume[1] / l_params.resolution[1], l_params.volume[2] / l_params.resolution[2] };
	
	float tsdf = NAN;
	int idx = 0;
	float prev_tsdf;
	
	int3 resolution = { convert_int(l_params.resolution[0]), convert_int(l_params.resolution[1]), convert_int(l_params.resolution[2]) };
	//int3 resolution = { (l_params.resolution[0]), (l_params.resolution[1]), (l_params.resolution[2]) };
	
	#ifdef MIROIR_GL_INTEROP
		int2 coord = {u, v};
		float4 color = {0.0, 0.0, 0.0, 1.0};
		write_imagef(frameBuffer, coord, color);
	#endif
	
	while(depth_march < l_params.volume[0])
	{
		vertex_4 += ray_step;
		depth_march += step;
		int3 voxel = getVoxel(vertex_4, l_params); 
		
		if( voxel.z < 0 || voxel.z >= resolution.z ||
			voxel.y < 0 || voxel.y >= resolution.y ||
			voxel.x < 0 || voxel.x >= resolution.x )
		{
			continue;
		}
		
		int tmp_idx = voxel.z + voxel.x * resolution.z + voxel.y * resolution.x * resolution.z;
		//idx = voxel.z + voxel.x * resolution.z + voxel.y * resolution.x * resolution.z;
		if(tmp_idx == idx)
			continue;
		idx = tmp_idx;
		
		prev_tsdf = tsdf;
		tsdf = F[idx].x;	// bottleneck
		
		if(isnan(tsdf) || isnan(prev_tsdf))
			continue;
		
		if(prev_tsdf < 0.0f && tsdf > 0.0f)
			break;
		
		if(prev_tsdf >= 0.0f && tsdf <= 0.0f)
		{	// zero crossing
			
			//float Ftdt = tsdf;
			float Ftdt = interpolateTrilinearly(F, vertex_4, voxel, cellsize, l_params);
			if(isnan(Ftdt))
				break;
				
			float4 prev_vertex = vertex_4 - ray_step;
			int3 prev_voxel = getVoxel(prev_vertex, l_params);
			//float Ft = prev_tsdf;
			float Ft = interpolateTrilinearly(F, prev_vertex, prev_voxel, cellsize, l_params);
			if(isnan(Ft))
				break;
				
			if(Ftdt > 0.0f && Ft > 0.0f)
				continue;
			if(Ftdt < 0.0f && Ft < 0.0f)
				break;
				
			float Ts = (depth_march-step) - step * Ft / (Ftdt - Ft);
			
			float4 v_res = ray_start + ray_dir * Ts;
			
			vmap[pix_idx].x = v_res.x;
			vmap[pix_idx].y = v_res.y;
			vmap[pix_idx].z = v_res.z;
			
			float4 t;
			float3 n;

			t = v_res;
			t.x += cellsize.x;
			float Fx1 = interpolateTrilinearly(F,t,getVoxel(t, l_params), cellsize, l_params);
	
			if(isnan(Fx1))
				break;
	
			t = v_res;
			t.x -= cellsize.x;
			float Fx2 = interpolateTrilinearly(F,t,getVoxel(t, l_params), cellsize, l_params);

			if(isnan(Fx2))
				break;
			
			n.x = (Fx2 - Fx1);// / (2 * cellsize.x);

			t = v_res;
			t.y += cellsize.y;
			float Fy1 = interpolateTrilinearly(F,t,getVoxel(t, l_params), cellsize, l_params);

			if(isnan(Fy1))
				break;
			
			t = v_res;
			t.y -= cellsize.y;
			float Fy2 = interpolateTrilinearly(F,t,getVoxel(t, l_params), cellsize, l_params);

			if(isnan(Fy2))
				break;
			
			n.y = (Fy2 - Fy1);// / (2 * cellsize.y);

			t = v_res;
			t.z += cellsize.z;
			float Fz1 = interpolateTrilinearly(F,t,getVoxel(t, l_params), cellsize, l_params);

			if(isnan(Fz1))
				break;
			
			t = v_res;
			t.z -= cellsize.z;
			float Fz2 = interpolateTrilinearly(F,t,getVoxel(t, l_params), cellsize, l_params);

			if(isnan(Fz2))
				break;
			
			n.z = (Fz2 - Fz1);// / (2 * cellsize.z);
			
			n = normalize(n);
			
			nmap[pix_idx] = n;
			
			// phong shading
			
			// float3 lightPos = {1.0, 10.0, -1.0};
			// float3 lightDir = lightPos - v_res;
			// float4 lightDiffuse = {0.5, 0.5, 0.5, 1.0};
			// float4 lightSpec = {0.0, 0.0, 0.0, 1.0};
			// float4 matDiffuse = lightDiffuse;
			// float4 matSpec = lightSpec;
			// float matShininess = 1.0;
			// float attenuation = 1.0;
			// lightDir = normalize(lightDir);
			// float NdotL = max(dot(n, lightDir), 0.0);
			// float3 eyeVec = {-ray_dir.x, -ray_dir.y, -ray_dir.z};
			// if(NdotL > 0.0)
			// {
				// color += lightDiffuse * matDiffuse * NdotL * attenuation;
				// float3 R = normalize( -reflect(lightDir, n) );
				// float specular = pow( max(dot(R, eyeVec), 0.0), matShininess );
				// color += lightSpec * matSpec * specular * attenuation;
			// }
			
			#ifdef MIROIR_GL_INTEROP
				color.x = n.x * 0.5 + 0.5f;
				color.y = n.y * 0.5 + 0.5f;
				color.z = n.z * 0.5 + 0.5f;
				write_imagef(frameBuffer, coord, color);
			#endif
			
			return;
		}
	}
	
	vmap[pix_idx] = NAN;
	nmap[pix_idx] = NAN;
	
}
