#ifndef MIROIR_UTIL_H
#define MIROIR_UTIL_H

#include <CL/cl.h>

namespace MiroirUtil
{
	inline float angleFromRotMatrix(const cl_float3 rot[3])
	{
		return std::atan( std::sqrt((rot[2].s[1] - rot[1].s[2])*(rot[2].s[1] - rot[1].s[2]) + (rot[0].s[2] - rot[2].s[0])*(rot[0].s[2] - rot[2].s[0]) + (rot[1].s[0] - rot[0].s[1])*(rot[1].s[0] - rot[0].s[1])) 
			/ (rot[0].s[0] + rot[1].s[1] + rot[2].s[2] - 1.0f) );
	}

	inline void invertCamTransform( const cl_float4 cam_pose[4], cl_float4 cam_pose_inv[4] )
	{
		// transpose rotation matrix
		cam_pose_inv[0].s[0] = cam_pose[0].s[0];
		cam_pose_inv[0].s[1] = cam_pose[1].s[0];
		cam_pose_inv[0].s[2] = cam_pose[2].s[0];
		cam_pose_inv[0].s[3] = 0.0f;

		cam_pose_inv[1].s[0] = cam_pose[0].s[1];
		cam_pose_inv[1].s[1] = cam_pose[1].s[1];
		cam_pose_inv[1].s[2] = cam_pose[2].s[1];
		cam_pose_inv[1].s[3] = 0.0f;

		cam_pose_inv[2].s[0] = cam_pose[0].s[2];
		cam_pose_inv[2].s[1] = cam_pose[1].s[2];
		cam_pose_inv[2].s[2] = cam_pose[0].s[0];
		cam_pose_inv[2].s[3] = 0.0f;

		// negate translation vector and multiply by rotation inverse
		/*cam_pose_inv[3].s[0] = -(cam_pose[3].s[0] * cam_pose_inv[0].s[0] + cam_pose[3].s[1] * cam_pose_inv[1].s[0] + cam_pose[3].s[2] * cam_pose_inv[2].s[0]);
		cam_pose_inv[3].s[1] = -(cam_pose[3].s[0] * cam_pose_inv[0].s[1] + cam_pose[3].s[1] * cam_pose_inv[1].s[1] + cam_pose[3].s[2] * cam_pose_inv[2].s[1]);
		cam_pose_inv[3].s[2] = -(cam_pose[3].s[0] * cam_pose_inv[0].s[2] + cam_pose[3].s[1] * cam_pose_inv[1].s[2] + cam_pose[3].s[2] * cam_pose_inv[2].s[2]);
		cam_pose_inv[3].s[3] = 1.0f;*/
		cam_pose_inv[3].s[0] = -(cam_pose[3].s[0]);
		cam_pose_inv[3].s[1] = -(cam_pose[3].s[1]);
		cam_pose_inv[3].s[2] = -(cam_pose[3].s[2]);
		cam_pose_inv[3].s[3] = 1.0f;
	}

	inline void mat3Mult(const cl_float3 A[3], const cl_float3 B[3], cl_float3 res[3])
	{
		res[0].s[0] = A[0].s[0] * B[0].s[0] + A[1].s[0] * B[0].s[1] + A[2].s[0] * B[0].s[2];
		res[0].s[1] = A[0].s[1] * B[0].s[0] + A[1].s[1] * B[0].s[1] + A[2].s[1] * B[0].s[2];
		res[0].s[2] = A[0].s[2] * B[0].s[0] + A[1].s[2] * B[0].s[1] + A[2].s[2] * B[0].s[2];

		res[1].s[0] = A[0].s[0] * B[1].s[0] + A[1].s[0] * B[1].s[1] + A[2].s[0] * B[1].s[2];
		res[1].s[1] = A[0].s[1] * B[1].s[0] + A[1].s[1] * B[1].s[1] + A[2].s[1] * B[1].s[2];
		res[1].s[2] = A[0].s[2] * B[1].s[0] + A[1].s[2] * B[1].s[1] + A[2].s[2] * B[1].s[2];

		res[2].s[0] = A[0].s[0] * B[2].s[0] + A[1].s[0] * B[2].s[1] + A[2].s[0] * B[2].s[2];
		res[2].s[1] = A[0].s[1] * B[2].s[0] + A[1].s[1] * B[2].s[1] + A[2].s[1] * B[2].s[2];
		res[2].s[2] = A[0].s[2] * B[2].s[0] + A[1].s[2] * B[2].s[1] + A[2].s[2] * B[2].s[2];
	}

	inline void mat4Mult( const cl_float4 A[4], const cl_float4 B[4], cl_float4 res[4])
	{
		res[0].s[0] = A[0].s[0] * B[0].s[0] + A[1].s[0] * B[0].s[1] + A[2].s[0] * B[0].s[2] + A[3].s[0] * B[0].s[3];
		res[0].s[1] = A[0].s[1] * B[0].s[0] + A[1].s[1] * B[0].s[1] + A[2].s[1] * B[0].s[2] + A[3].s[1] * B[0].s[3];
		res[0].s[2] = A[0].s[2] * B[0].s[0] + A[1].s[2] * B[0].s[1] + A[2].s[2] * B[0].s[2] + A[3].s[2] * B[0].s[3];;
		res[0].s[3] = A[0].s[3] * B[0].s[0] + A[1].s[3] * B[0].s[1] + A[2].s[3] * B[0].s[2] + A[3].s[3] * B[0].s[3];

		res[1].s[0] = A[0].s[0] * B[1].s[0] + A[1].s[0] * B[1].s[1] + A[2].s[0] * B[1].s[2] + A[3].s[0] * B[1].s[3];
		res[1].s[1] = A[0].s[1] * B[1].s[0] + A[1].s[1] * B[1].s[1] + A[2].s[1] * B[1].s[2] + A[3].s[1] * B[1].s[3];
		res[1].s[2] = A[0].s[2] * B[1].s[0] + A[1].s[2] * B[1].s[1] + A[2].s[2] * B[1].s[2] + A[3].s[2] * B[1].s[3];
		res[1].s[3] = A[0].s[3] * B[1].s[0] + A[1].s[3] * B[1].s[1] + A[2].s[3] * B[1].s[2] + A[3].s[3] * B[1].s[3];

		res[2].s[0] = A[0].s[0] * B[2].s[0] + A[1].s[0] * B[2].s[1] + A[2].s[0] * B[2].s[2] + A[3].s[0] * B[0].s[3];
		res[2].s[1] = A[0].s[1] * B[2].s[0] + A[1].s[1] * B[2].s[1] + A[2].s[1] * B[2].s[2] + A[3].s[1] * B[0].s[3];
		res[2].s[2] = A[0].s[2] * B[2].s[0] + A[1].s[2] * B[2].s[1] + A[2].s[2] * B[2].s[2] + A[3].s[2] * B[0].s[3];
		res[2].s[3] = A[0].s[3] * B[2].s[0] + A[1].s[3] * B[2].s[1] + A[2].s[3] * B[2].s[2] + A[3].s[3] * B[0].s[3];

		res[3].s[0] = A[0].s[0] * B[3].s[0] + A[1].s[0] * B[3].s[1] + A[2].s[0] * B[3].s[2] + A[3].s[0] * B[3].s[3];
		res[3].s[1] = A[0].s[1] * B[3].s[0] + A[1].s[1] * B[3].s[1] + A[2].s[1] * B[3].s[2] + A[3].s[1] * B[3].s[3];
		res[3].s[2] = A[0].s[2] * B[3].s[0] + A[1].s[2] * B[3].s[1] + A[2].s[2] * B[3].s[2] + A[3].s[2] * B[3].s[3];
		res[3].s[3] = A[0].s[3] * B[3].s[0] + A[1].s[3] * B[3].s[1] + A[2].s[3] * B[3].s[2] + A[3].s[3] * B[3].s[3];
	}

	//Cholesky decomposition of matrix A
	inline void cholesky(int n, cl_float* A, cl_float* decomp)
	{
		/*double sum1 = 0.0;
		double sum2 = 0.0;
		double sum3 = 0.0;
	
		decomp[0] = sqrt(A[0]);
		for (int j = 1; j <= n-1; j++)
			decomp[j] = A[j]/decomp[0];

		for (int i = 1; i <= (n-2); i++)
		{
			for (int k = 0; k <= (i-1); k++)
				sum1 += pow(decomp[i * n + k], 2);
			decomp[i * n + i]= sqrt(A[i * n + i]-sum1);
			for (int j = (i+1); j <= (n-1); j++)
			{
				for (int k = 0; k <= (i-1); k++)
					sum2 += decomp[j * n + k]*decomp[i * n + k];
				decomp[j * n + i]= (A[j * n + i]-sum2)/decomp[i * n + i];
			}
		}
		for (int k = 0; k <= (n-2); k++)
			sum3 += pow(decomp[(n-1)* n +k], 2);
		decomp[(n-1)* n + n-1] = sqrt(A[(n-1)* n + n-1]-sum3);*/

		for (int i = 0; i < n; i++)
			for (int j = 0; j < (i+1); j++) {
				double s = 0;
				for (int k = 0; k < j; k++)
					s += decomp[i * n + k] * decomp[j * n + k];
				decomp[i * n + j] = (i == j) ? sqrt(A[i * n + i] - s) : (1.0 / decomp[j * n + j] * (A[i * n + j] - s));
			}
	}

	inline double determinant(cl_float* A, int n)
	{
		/* const int dim  is the dimensionality of the matrix */
		double det = 0.0;
		for (int i = 0; i < n; i++)
		{
			double a = 1.0, b = 1.0;
			for (int row = 0; row < n; row++)
			{
				a *= A[row*n+((i+row)%n)];
				b *= A[row*n+((n-1) - (i+row)%n)];
			}
			det += a - b;
		}
		return det;
	}
}

#endif