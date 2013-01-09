
/**
 * This kernel performs bilateral filtering of the depth map and outputs the vertex map and normal map
*/

__kernel void measurement_vertices(	__global const ushort* src, 
								__global const int* width, 
								__global const int* height,
								__global const float* fl,		// focal length
								//__global ushort* dst,
								__global float3* vmap,
								__global const float* sigma_s, 
								__global const float* sigma_r
						)
{
   /* get_global_id(0) returns the ID of the thread in execution.
   As many threads are launched at the same time, executing the same kernel,
   each one will receive a different ID, and consequently perform a different computation.*/
  
   const int x = get_global_id(0);
   const int y = get_global_id(1);
   const float s = *sigma_s;
   const float r = *sigma_r;
   
   const int radius = 8;
   int w = *width;	// buffer into shared memory?
   int h = *height;
   float focal_length = *fl;
   
   int tlx = max(x-radius, 0);
   int tly = max(y-radius, 0);
   int brx = min(x+radius, w);
   int bry = min(y+radius, h);
   
   int idx = w * y + x;
   //depth_l[idx] = src[idx];
   float sum = 0;
   float wp = 0;	// normalizing constant
   
	//barrier(CLK_LOCAL_MEM_FENCE);
   float src_depth = src[idx];
   
   float s2 = s*s;
   float r2 = r*r;
   
   if(src_depth != 0)
   {
	   for(int i=tlx; i< brx; i++)
	   {
			for(int j=tly; j<bry; j++)
			{
			// cost: 
				float delta_dist = (float)((x - i) * (x - i) + (y - j) * (y - j));
					
				int idx2 = w * j + i;
				float d = src[idx2]; // cost: 0.013s	// TODO : use shared memory?
				float delta_depth = (src_depth - d) * (src_depth - d); 
				float weight = native_exp( -(delta_dist / s2 + delta_depth / r2) ); //cost : 
				sum += weight * d;
				wp += weight;
			}
	   } 
	   float res = sum / wp;
	   //dst[idx] = res;
	   vmap[idx].x = res*(x-w/2)/focal_length;
	   vmap[idx].y = res*(y-h/2)/focal_length;
	   vmap[idx].z = res;
	   //vmap[idx].x = res*(x-(*pptu))/(*fl);
		//vmap[idx].y = res*(y-(*pptv))/(*fl);
   }
   else
   {
		//dst[idx] = NAN;
	   vmap[idx].x = NAN;
	   vmap[idx].y = NAN;
	   vmap[idx].z = NAN;
   }
}

__kernel void measurement_normals(	
	__global const int* width,
	__global const int* height,
	__global const float3* vmap,
	__global float3* nmap
)
{  
	const int x = get_global_id(0);
    const int y = get_global_id(1);
	int w = *width;
	int h = *height;
	
	int idx = w * y + x;
	
   // // normal map compute
   float3 v1 = (x < (w-1))?vmap[w * y + x + 1]:vmap[w * y + x - 1];
   float3 v2 = (y < (h-1))?vmap[w * (y + 1) + x]:vmap[w * (y - 1) + x];
   
   //float3 normal_nan = {0,0,0};
   float3 v = vmap[idx];
   nmap[idx] = normalize( ( isnan(v1) || isnan(v2) || isnan(v) ) ? NAN : cross(v1 - v,  v2 - v));
}
