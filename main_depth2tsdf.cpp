#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include <Eigen/Dense>

class OrthoDepthMap
{
public:
	OrthoDepthMap() : width_(0), height_(0), dx_(0.0f), dy_(0.0f), x0_(0), y0_(0)
	{}

	bool readDepthMap(const char* filename)
	{
		std::cout << "- Reading depth data : " << filename << "..." << std::flush;

		std::ifstream inFile(filename, std::ios::binary);

		if (!inFile.is_open())
		{
			std::cout << "Error, can't open " << filename << " for reading!" << std::endl;
			return false;
		}

		inFile.read((char*)&width_, sizeof(int));
		inFile.read((char*)&height_, sizeof(int));
		inFile.read((char*)&dx_, sizeof(float));
		inFile.read((char*)&dy_, sizeof(float));
		inFile.read((char*)&x0_, sizeof(float));
		inFile.read((char*)&y0_, sizeof(float));
		inFile.read((char*)&z_img_, sizeof(float));

		depthData_.resize(width_*height_);

		inFile.read((char*)depthData_.data(), sizeof(float)*width_*height_);

		//std::cout << depthData_[width_*height_ / 2] << std::endl;

		inFile.close();

		std::cout << "done" << std::endl;

		return true;
	}

	void printInfo()
	{
		std::cout << "- Depth data : " << "\n"
			<< " . (nx, ny) = (" << width_ << " x " << height_ << ")" << "\n"
			<< " . (dx, dy) = (" << dx_ << ", " << dy_ << ")" << "\n"
			<< " . (x0, y0) = (" << x0_ << ", " << y0_ << ")" << "\n"
			<< " . image plane : z=" << z_img_ << std::endl;
	}

	float bilinear_interpolation(float* f, int index, int w, float s, float t)
	{
		float a[4];
		a[0] = f[index];
		a[1] = f[index + 1];
		a[2] = f[index + w];
		a[3] = f[index + w + 1];

		auto  result = std::minmax_element(a, a + 4);

		float fmin = *result.first;
		float fmax = *result.second;

		if (fmin == 0 && fmax > 0)
			return fmax;
		else
			return a[0] * (1.0 - s)*(1.0 - t) + a[1] * s*(1.0 - t) + a[2] * (1.0 - s)*t + a[3] * s*t;
	}

	void computeTSDF(int gridNx, int gridNy, int gridNz, int truncSize, float* tsdf)
	{
		std::cout << "- Computing TSDF : " << std::flush;

		// Compute the depth range for the given data
		float zmin, zmax;
		zmin = 1.e34f;
		zmax = -1.0f;
		for (int i = 0; i < width_*height_; ++i)
		{
			if (depthData_[i] > 0.0f)
			{
				if (depthData_[i] > zmax) zmax = depthData_[i];
				if (depthData_[i] < zmin) zmin = depthData_[i];
			}
		}

		std::cout << " depth range : [" << zmin << ", " << zmax << "]" << std::endl;

		zmin += z_img_;
		zmax += z_img_;

		float Lz = zmax - zmin;
		
		zmin -= Lz * 0.05f;
		zmax += Lz * 0.05f;
		Lz *= 1.1f;

		zmin_ = zmin;
		zmax_ = zmax;

		// Set the volume data
		float Lx = dx_ * width_;
		float Ly = dy_ * height_;

		float gridDx = Lx / gridNx;
		float gridDy = Ly / gridNy;
		float gridDz = Lz / gridNz;

		Eigen::Vector3f bmin, bmax;
		bmin = Eigen::Vector3f(x0_-0.5f*dx_, y0_-0.5f*dy_, zmin);
		bmax = Eigen::Vector3f(Lx, Ly, Lz) + bmin;

		Eigen::Vector3f blockCenter = 0.5f*(bmin + bmax);

		//std::cout << "- Bounding box : (" << bmin.transpose() << ")~(" << bmax.transpose() << ")" << std::endl;

		std::cout << "- Volume grid Info :\n"
			<< " . center       : (" << blockCenter.transpose() << ")\n"
			<< " . size         : " << gridNx << "x" << gridNy << "x" << gridNz << ")\n"
			<< " . spacing      : (" << gridDx << ", " << gridDy << ", " << gridDz << ")\n" 
			<< " . bounding box : (" << bmin.transpose() << ")~(" << bmax.transpose() << ")" 
			<< std::endl;


		// Compute the truncated signed distance funtions on the voxel centers
		float INF = std::numeric_limits<float>::infinity();
		float eps = std::numeric_limits<float>::epsilon();

		float truncDist = gridDz * gridNz * truncSize;

		std::fill(tsdf, tsdf + gridNx * gridNy * gridNz, -1.0f);

		Eigen::Vector3f p;
		Eigen::Vector3f voxel(gridDx, gridDy, gridDz);
		unsigned int count = 0, count1 = 0, count2 = 0;

		int width = width_;
		int height = height_;

		for (int k = 0; k < gridNz; ++k)
		{
			for (int j = 0; j < gridNy; ++j)
			{
				for (int i = 0; i < gridNx; ++i)
				{
					// voxel center
					p = Eigen::Vector3f(i * gridDx, j * gridDy, k * gridDz) + bmin + 0.5f*voxel;

					float cx = (p(0) - bmin(0)) / dx_;
					float cy = (p(1) - bmin(1)) / dy_;

					int u, v;
					u = (int)floor(cx);
					v = (int)floor(cy);

					float z = p(2)-z_img_;

					if (z < INF && z > -INF && u > 0 && u < width - 1 && v > 0 && v < height - 1
						&& depthData_[v*width+u] > 0.0 && depthData_[v*width + u] < INF)
					{
						// NOTE : We are using bilinear interpolation for smoother depthmap
						float du = cx - (float)u;
						float dv = cy - (float)v;

						float depth = bilinear_interpolation(depthData_.data(), v*width + u, width, du, dv);

						if (fabs(z-depth) < truncDist)
						{
							float f = (z-depth) / truncDist; // This guarantees |f|<=1

							tsdf[k * gridNx * gridNy + j * gridNx + i] = f;
						}
						/*else
						{
							if (depth - z > truncDist)
								tsdf[k* gridNx * gridNy + j * gridNx + i] = 1.0f;
						}*/
					}
					else
					{
						if(depthData_[v*width+u] <= 0.0f )	
							tsdf[k* gridNx * gridNy + j * gridNx + i] = 1.0f;
						//count++;
					}
				}// end for(i)
			}// end for(j)
		}// end for(k)

		//std::cout << "- # of voxels with  improper truncated signed distances =" << count << " " << count1 << std::endl;
	}

	bool writeTSDF(const char* filename, int gnx, int gny, int gnz, float* f)
	{
		std::cout << "- Writing TSDF data : " << filename << "..." << std::flush;

		std::ofstream outFile(filename, std::ios::binary);
		if (!outFile.is_open())
		{
			std::cout << "Error, can't open " << filename << " for writing!" << std::endl;
			return false;
		}

		Eigen::Vector3f bmin, bmax;
		bmin = Eigen::Vector3f(x0_ - 0.5f*dx_, y0_ - 0.5f*dy_, zmin_);
		bmax = Eigen::Vector3f(dx_*width_, dy_*height_, zmax_-zmin_) + bmin;

		outFile.write((char*)bmin.data(), sizeof(float) * 3);
		outFile.write((char*)bmax.data(), sizeof(float) * 3);

		outFile.write((char*)&gnx, sizeof(unsigned int));
		outFile.write((char*)&gny, sizeof(unsigned int));
		outFile.write((char*)&gnz, sizeof(unsigned int));

		outFile.write((char*)f, sizeof(float)*gnx*gny*gnz);

		outFile.close();
		std::cout << "done" << std::endl;

		return true;
	}

public:
	int width_;
	int height_;

	float dx_, dy_;
	float x0_, y0_;

	float zmin_, zmax_;
	float z_img_;

	std::vector<float> depthData_;
};

int main(int argc, char** argv)
{
	if (argc <= 5)
	{
		std::cout << "- Usage : depth2tsdf_ortho.exe <depth_data> <nx> <ny> <nz> <trunc_size>" << std::endl;
		return -1;
	}

	const char* depthFile = argv[1];
	const int   nx = atoi(argv[2]);
	const int   ny = atoi(argv[3]);
	const int   nz = atoi(argv[3]);
	const int	truncSize = atoi(argv[4]);

	OrthoDepthMap depthMap;
	depthMap.readDepthMap(depthFile);
	depthMap.printInfo();

	std::vector<float> tsdf(nx*ny*nz);
	depthMap.computeTSDF(nx, ny, nz, truncSize, tsdf.data());

	depthMap.writeTSDF("tsdf.dat", nx, ny, nz, tsdf.data());

	return 0;
}