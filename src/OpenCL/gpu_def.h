#ifndef GPU_DEF 
#define GPU_DEF

#define SHORT_NAN 32767
#define SHORT_MAX 32766

#define GL_INTEROP // Caution!! Need to clear ComputeCache if uncommented

struct TsdfParams
{
    float resolution[3];
    float volume[3];
    float mu;
};

#ifdef _GPU_

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

// m is column wise, i.e m[0] is column 0, m[1] column 1, etc.
bool _isnan3(float3 v)
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

bool _isnan4(float4 v)
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w);
}

bool _isnan2(float2 v)
{
	return isnan(v.x) || isnan(v.y);
}

// short between -32767 and 32767
void pack_tsdf(float val, float weight, __global short2* tsdf, int idx)
{
	//tsdf[idx].x = convert_short(val * convert_float(SHORT_MAX));
	//tsdf[idx].y = convert_short(weight)-SHORT_MAX;
	short2 res = {convert_short(val * convert_float(SHORT_MAX)), convert_short(weight)-SHORT_MAX};
	tsdf[idx] = res;
}

float2 unpack_tsdf(short2 tsdf)
{
	float2 res = {convert_float(tsdf.x) / convert_float(SHORT_MAX), convert_float(tsdf.y)+convert_float(SHORT_MAX)};
	return res;
}

float4 mul( float4 m[4], float4 v )
{
	float4 res = {  m[0].x*v.x + m[1].x*v.y + m[2].x*v.z + m[3].x*v.w,
					m[0].y*v.x + m[1].y*v.y + m[2].y*v.z + m[3].y*v.w,
					m[0].z*v.x + m[1].z*v.y + m[2].z*v.z + m[3].z*v.w,
					m[0].w*v.x + m[1].w*v.y + m[2].w*v.z + m[3].w*v.w };
	return res;
}

float3 mul_homogenize( float4 m[4], float3 v )
{
	float3 res = {  m[0].x*v.x + m[1].x*v.y + m[2].x*v.z + m[3].x*1.0f,
					m[0].y*v.x + m[1].y*v.y + m[2].y*v.z + m[3].y*1.0f,
					m[0].z*v.x + m[1].z*v.y + m[2].z*v.z + m[3].z*1.0f
				};
	return res;
}

float4 getVoxelGlobal( float voxel_x, float voxel_y, float voxel_z, float3 cellsize )
{
	float4 voxel = {(voxel_x + 0.5f) * cellsize.x, (voxel_y + 0.5f) * cellsize.y, (voxel_z + 0.5f) * cellsize.z, 1.0f };
	return voxel;
}

bool isNull(float3 v)
{
	return v.x == 0 && v.y == 0 && v.z == 0;
}

// bool isNull(float4 v)
// {
	// return v.x == 0 && v.y == 0 && v.z == 0 && v.z == 0;
// }

#endif
#endif