#ifndef WQBALLGPU_H
#define WQBALLGPU_H

// External lib headers
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

// C++ headers
#include <utility>

// OW core headers
#include "core/kernel/WKernel.h"
#include "core/common/WObjectNDIP.h"
#include "core/common/WProperties.h"
#include "core/dataHandler/WFiberAccumulator.h"
#include "core/dataHandler/WDataSetRawHARDI.h"

#include "WQBallAlgorithm.h"

class WQBallGPU: public WObjectNDIP< WQBallAlgorithm >
{
public:
	WQBallGPU();
	virtual ~WQBallGPU();

	virtual void prepareAlgorithm( boost::shared_ptr< WDataSetRawHARDI > datasetRaw, boost::shared_ptr< WDataSetScalar > datasetScalar, boost::filesystem::path path, boost::shared_ptr< WProgress > prog );
	virtual std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > operator()( boost::shared_ptr< WProgress > prog, WPosition startPpos );
	virtual void cleanUp();

private:

	WPropInt m_seedPoints;
    WPropInt m_seedsPerVoxel;
    WPropDouble m_stepSize;
    WPropInt m_stepDistance;

	WPropDouble m_fa;
	WPropDouble m_curv;

	WPropDouble m_percentRange;
	WPropInt m_degreeRange;

    WPropGroup m_sh;
    WPropGroup m_fibers;
    WPropGroup m_thresholds;
    WPropGroup m_odfThresh;

    size_t global;
    size_t local;

	// variables for OpenCL kernel
	FILE *m_f;

	// data arrays for opencl mem
	short* m_hardiData;
	float* m_scalarDataIn;

	//constant data
	float* b_Odf;
	float* b_Base;
	float* b_Coeff;
	float* b_H;
	cl_float4* b_Reconstruction;
	float* b_Properties;
	ushort* b_Nzg;
	cl_ushort4 b_GridDimension;
	uint* b_Dimensions;
	cl_int4 b_StartPos;
	short *b_Transformation;
	ushort *b_Neighbors;
	cl_float4 *b_OldDirection;

	// OpenCL
	cl_int m_err;
	cl_platform_id m_platform;
	cl_device_id m_device;
	cl_uint m_num_platforms;
	cl_uint m_num_devices;
	cl_context m_context;
	cl_command_queue m_command_queue;
	cl_program m_program;

	cl_mem m_clHardi;
	cl_mem m_clScalars;
	cl_mem m_clFibers;
	cl_mem m_clScalarsOut;
	cl_mem m_clOdf;
	cl_mem m_clBase;
	cl_mem m_clCoeff;
	cl_mem m_clH;
	cl_mem m_clReconstruction;
	cl_mem m_clProperties;
	cl_mem m_clNzg;
	cl_mem m_clGridDimension;
	cl_mem m_clDimensions;
	cl_mem m_clStartPos;
	cl_mem m_clPositionArray;
	cl_mem m_clSampleArray;
	cl_mem m_clOdfArray;
	cl_mem m_clPeaksArray;
	cl_mem m_clNeighbors;
	cl_mem m_clMainDir;
	cl_mem m_clOldDirection;
	//cl_mem m_clTransformation;

	cl_mem m_clTest;

	int initOpenCL();
	void createDataArrays();
	void releaseMemory();

	int getPot( int a ) {
		int count = 1;
		while( count < a ) count *= 2;
		return count;
	}
	
};

#endif

