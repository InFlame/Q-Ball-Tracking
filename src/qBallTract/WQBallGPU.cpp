#include <utility>

// OW core headers
#include "core/kernel/WKernel.h"

#include "WQBallGPU.h"
#include "WQBallAlgorithm.h"

#define MAX_SOURCE_SIZE ( 0x100000 )

WQBallGPU::WQBallGPU():WObjectNDIP< WQBallAlgorithm >( "GPU", "QBall Tractography on GPU") {

	m_ready = false;


	// properies
	m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

	// fibers
    m_seedPoints = m_properties->addProperty( "Seed points", "Choose number of seed points", 1000, m_propCondition );
    m_seedsPerVoxel = m_properties->addProperty( "Points per Voxel", "Seed points per Voxel", 1, m_propCondition );
    m_stepSize = m_properties->addProperty( "Step size", "Distance for Fibers", 0.5, m_propCondition );
    m_stepDistance = m_properties->addProperty( "Distance", "Step distance", 500, m_propCondition );

    // thresholds
    m_fa = m_properties->addProperty( "FA", "Min Fractional anisotropy", 0.05, m_propCondition );
    m_curv = m_properties->addProperty( "Curvature", "Max Curvature", 75.0, m_propCondition );

    // odf thresholds
    m_percentRange = m_properties->addProperty( "% range", "accepted magnitudes", 0.33, m_propCondition );
    m_percentRange->setMin( 0.0 );
    m_percentRange->setMax( 1.0 );
    m_degreeRange = m_properties->addProperty( "deg range", "accepted degrees", 45, m_propCondition );

    m_fibers = m_properties->addPropertyGroup( "Fibers", "Fiber options" );
    	m_fibers->addProperty( m_seedPoints );
    	m_fibers->addProperty( m_seedsPerVoxel );
    	m_fibers->addProperty( m_stepSize );
    m_thresholds = m_properties->addPropertyGroup( "Thresholds", "Conditions for stopping a fiber" );
    	m_thresholds->addProperty( m_fa );
    	m_thresholds->addProperty( m_curv );
    m_odfThresh = m_properties->addPropertyGroup( "Odf", "Options for ODF" );
    	m_odfThresh->addProperty( m_percentRange );
    	m_odfThresh->addProperty( m_degreeRange );

	m_ready = true;
}

WQBallGPU::~WQBallGPU() {}

int WQBallGPU::initOpenCL() {

	m_platform = NULL;
	m_device = NULL;
	m_err |= clGetPlatformIDs( 1, &m_platform, &m_num_platforms );
	m_err |= clGetDeviceIDs( m_platform, CL_DEVICE_TYPE_GPU, 1, &m_device, &m_num_devices );

	m_context = clCreateContext( NULL, 1, &m_device, NULL, NULL, &m_err );

	m_command_queue = clCreateCommandQueue( m_context, m_device, 0, &m_err );

	//m_f = fopen( ( m_path / "opencl" / "testkernel.cl" ).string().c_str(), "r" );
	m_f = fopen( "/u/mai11dre/QBallTractToolbox/src/qBallTract/OpenCL/QBall.cl", "r" );
	wlog::debug( "GPU" ) << m_path.string();
	if( !m_f ) {
		wlog::debug( "GPU" ) << "Failed to load kernel from file";
		throw WPreconditionNotMet( std::string( "Could not load opencl source code" ) );		
	}
	char* programSource = ( char* )malloc( MAX_SOURCE_SIZE );
	size_t sourceSize = fread( programSource, 1, MAX_SOURCE_SIZE, m_f );

	fclose( m_f );


	m_program = clCreateProgramWithSource( m_context, 1, ( const char** ) &programSource, ( const size_t * ) &sourceSize, &m_err );

}

void WQBallGPU::prepareAlgorithm( boost::shared_ptr< WDataSetRawHARDI > datasetRaw, boost::shared_ptr< WDataSetScalar > datasetScalar, boost::filesystem::path path, boost::shared_ptr< WProgress > prog ) {
	WQBallAlgorithm::prepareAlgorithm( datasetRaw, datasetScalar, path, prog );

	initOpenCL();

	m_hardiData = new short[ datasetRaw->getValueSet()->dimension() * m_grid->size() ];
	m_scalarDataIn = new float[ m_gridScalar->size() ];

	boost::shared_ptr< WValueSet< short > > rawData = boost::dynamic_pointer_cast< WValueSet< short > >( datasetRaw->getValueSet() );
	boost::shared_ptr< WValueSet< float > > rawScalar = boost::dynamic_pointer_cast< WValueSet< float > >( datasetScalar->getValueSet() );
	memcpy( m_hardiData, rawData->rawData(), sizeof( short ) * datasetRaw->getValueSet()->dimension() * m_grid->size() );
	memcpy( m_scalarDataIn, rawScalar->rawData(), sizeof( float ) * m_gridScalar->size() );

	global = 512;
	local = 32;

	size_t seedPoints = m_seedPoints->get( true );
	size_t steps = m_stepDistance->get ( true ) / m_stepSize->get( true );
	size_t vertices = seedPoints * steps;

	/* !!! peaks array ersetzen: im kernel getMaxima direkt hautrichtungen eintragen */

	m_clHardi = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( short ) * datasetRaw->getValueSet()->dimension() * m_grid->size(), NULL, &m_err );
	m_clScalars = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( float ) * m_gridScalar->size(), NULL, &m_err );
	m_clFibers = clCreateBuffer( m_context, CL_MEM_WRITE_ONLY, sizeof( float ) * vertices * global, NULL, &m_err );
	m_clScalarsOut = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( float ) * m_gridScalar->size(), NULL, &m_err );
	m_clOdf = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( float ) * m_odf.getNbCols() * m_odf.getNbRows(), NULL, &m_err );
	m_clBase = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( float ) * m_baseMatrix.getNbCols() * m_baseMatrix.getNbRows(), NULL, &m_err );
	m_clCoeff = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( float ) * m_coeffBase.getNbCols() * m_coeffBase.getNbRows(), NULL, &m_err );
	m_clH = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( float ) * m_h.getNbCols() * m_h.getNbRows(), NULL, &m_err );
	m_clReconstruction = clCreateBuffer( m_context, CL_MEM_READ_ONLY, 4 * sizeof( float ) * m_reconstructionPoints.size(), NULL, &m_err );
	m_clProperties = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( float ) * 6, NULL, &m_err );
	m_clNzg = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( ushort ) * m_nonZeroGradients.size(), NULL, &m_err );
	m_clGridDimension = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( ushort ) * 4, NULL, &m_err );
	m_clDimensions = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( uint ) * 6, NULL, &m_err );
	m_clStartPos = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( int ) * 4, NULL, &m_err );
	m_clPositionArray = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( int ) * 4 * seedPoints, NULL, &m_err );
	m_clSampleArray = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( float ) * datasetRaw->getNonZeroGradientIndexes().size() * global, NULL, &m_err );
	m_clOdfArray = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( float ) * m_reconstructionPoints.size() * global, NULL, &m_err );
	m_clTest = clCreateBuffer( m_context, CL_MEM_WRITE_ONLY, sizeof( float ) * global, NULL, &m_err );
	m_clPeaksArray = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( short ) * m_reconstructionPoints.size() * global, NULL, &m_err );
	m_clNeighbors = clCreateBuffer( m_context, CL_MEM_READ_ONLY, sizeof( ushort ) * m_neighbors.size() * 6, NULL, &m_err );
	m_clMainDir = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( float ) * 4 * 12 * global, NULL, &m_err );
	m_clOldDirection = clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof( float ) * 4 * global, NULL, &m_err );
	wlog::debug( "GPU" ) << "Buffer created. Errorcode: " << m_err;

}

std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > WQBallGPU::operator()( boost::shared_ptr< WProgress > progress, WPosition startPosition ) {

	// kommt später wieder weg, zum debuggen da
	if( clBuildProgram( m_program, m_num_devices, &m_device, NULL, NULL, NULL ) != CL_SUCCESS ) {
		printf("Error building program\n");

		char buffer[4096];
		size_t length;

		clGetProgramBuildInfo(
			m_program,
			m_device,
			CL_PROGRAM_BUILD_LOG,
			sizeof(buffer),
			buffer,
			&length
			);

		printf("%s\n", buffer);
	} else wlog::debug( "GPU" ) << "Program successfully build!";

	//initOpenCL();
	createDataArrays();
	size_t seedPoints = m_seedPoints->get( true );
	size_t steps = m_stepDistance->get ( true ) / m_stepSize->get( true );
	size_t vertices = seedPoints * steps;

	float *fiberOut = new float[ vertices * global ];
	float *scalarOut = new float[ m_gridScalar->size() ];

	cl_int4 *positionArray = new cl_int4[ seedPoints ];

	float *sampleArray = new float[ datasetRaw->getNonZeroGradientIndexes().size() * global ];
	float *odfArray = new float[ m_reconstructionPoints.size() * global ];
	float *test = new float[ global ];

	short *peaksArray = new short[ m_reconstructionPoints.size() * global ];

	uint voxelIndex = m_grid->getVoxelNum( startPosition );

	cl_float4 *mainDir = new cl_float4[ 12 * global ];

	b_StartPos.s[0] = m_grid->getXVoxelCoord( startPosition );
	b_StartPos.s[1] = m_grid->getYVoxelCoord( startPosition );
	b_StartPos.s[2] = m_grid->getZVoxelCoord( startPosition );
	b_StartPos.s[3] = 1;

	for( int i=0; i<seedPoints; i++ ) {
		positionArray[i] = b_StartPos;
	}





	// copy params to memory buffer
	m_err = 0;
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clHardi, CL_TRUE, 0, sizeof( short ) * datasetRaw->getValueSet()->dimension() * m_grid->size(), m_hardiData, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clScalars, CL_TRUE, 0, sizeof( float ) * m_gridScalar->size(), m_scalarDataIn, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clOdf, CL_TRUE, 0, sizeof( float ) * m_odf.getNbRows() * m_odf.getNbCols(), b_Odf, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clBase, CL_TRUE, 0, sizeof( float ) * m_baseMatrix.getNbRows() * m_baseMatrix.getNbCols(), b_Base, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clCoeff, CL_TRUE, 0, sizeof( float ) * m_coeffBase.getNbRows() * m_coeffBase.getNbCols(), b_Coeff, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clH, CL_TRUE, 0, sizeof( float ) * m_h.getNbRows() * m_h.getNbCols(), b_H, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clReconstruction, CL_TRUE, 0, 4 * sizeof( float ) * m_reconstructionPoints.size(), b_Reconstruction, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clProperties, CL_TRUE, 0, sizeof( float ) * 6, b_Properties, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clNzg, CL_TRUE, 0, sizeof( ushort ) * m_nonZeroGradients.size(), b_Nzg, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clGridDimension, CL_TRUE, 0, sizeof( ushort ) * 4, &b_GridDimension, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clDimensions, CL_TRUE, 0, sizeof( uint ) * 6, b_Dimensions, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clStartPos, CL_TRUE, 0, sizeof( int ) * 4, &b_StartPos, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clPositionArray, CL_TRUE, 0, sizeof( int ) * 4 * seedPoints, positionArray, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clSampleArray, CL_TRUE, 0,  sizeof( float) * datasetRaw->getNonZeroGradientIndexes().size() * global, sampleArray, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clNeighbors, CL_TRUE, 0, sizeof( ushort ) * m_neighbors.size() * 6, b_Neighbors, 0, NULL, NULL );
	m_err |= clEnqueueWriteBuffer( m_command_queue, m_clOldDirection, CL_TRUE, 0, sizeof( float ) * 4 * global, b_OldDirection, 0, NULL, NULL );
	wlog::debug( "GPU" ) << "VRam allocated. Errorcode: " << m_err;


	/*cl_kernel kernel = clCreateKernel( m_program, "algorithm", &m_err );
	m_err |= clSetKernelArg( kernel, 0, sizeof( cl_mem ), ( void* ) &m_clHardi );
	m_err |= clSetKernelArg( kernel, 1, sizeof( cl_mem ), ( void* ) &m_clScalars );
	m_err |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), ( void* ) &m_clFibers );
	m_err |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), ( void* ) &m_clScalarsOut );
	m_err |= clSetKernelArg( kernel, 4, sizeof( cl_mem ), ( void* ) &m_clOdf );
	m_err |= clSetKernelArg( kernel, 5, sizeof( cl_mem ), ( void* ) &m_clBase );
	m_err |= clSetKernelArg( kernel, 6, sizeof( cl_mem ), ( void* ) &m_clCoeff );
	m_err |= clSetKernelArg( kernel, 7, sizeof( cl_mem ), ( void* ) &m_clH );
	m_err |= clSetKernelArg( kernel, 8, sizeof( cl_mem ), ( void* ) &m_clReconstruction );
	m_err |= clSetKernelArg( kernel, 9, sizeof( cl_mem ), ( void* ) &m_clProperties );
	m_err |= clSetKernelArg( kernel, 10, sizeof( cl_mem ), ( void* ) &m_clNzg );
	m_err |= clSetKernelArg( kernel, 11, sizeof( cl_mem ), ( void* ) &m_clGridDimension );
	m_err |= clSetKernelArg( kernel, 12, sizeof( cl_mem ), ( void* ) &m_clDimensions );
	m_err |= clSetKernelArg( kernel, 13, sizeof( cl_mem ), ( void* ) &m_clStartPos );

	global = m_gridScalar->size();
	local = 32;
	m_err |= clEnqueueNDRangeKernel( m_command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL );*/

	cl_event event;

	// draw sample
	m_err = 0;
	cl_kernel kernelDrawSample = clCreateKernel( m_program, "drawSample", &m_err );
	m_err |= clSetKernelArg( kernelDrawSample, 0, sizeof( cl_mem ), ( void* ) &m_clHardi );
	m_err |= clSetKernelArg( kernelDrawSample, 1, sizeof( cl_mem ), ( void* ) &m_clNzg );
	m_err |= clSetKernelArg( kernelDrawSample, 2, sizeof( cl_mem ), ( void* ) &m_clBase );
	m_err |= clSetKernelArg( kernelDrawSample, 3, sizeof( cl_mem ), ( void* ) &m_clCoeff );
	m_err |= clSetKernelArg( kernelDrawSample, 4, sizeof( cl_mem ), ( void* ) &m_clH );
	m_err |= clSetKernelArg( kernelDrawSample, 5, sizeof( cl_mem ), ( void* ) &m_clDimensions );
	m_err |= clSetKernelArg( kernelDrawSample, 6, sizeof( cl_mem ), ( void* ) &m_clPositionArray );
	m_err |= clSetKernelArg( kernelDrawSample, 7, sizeof( cl_mem ), ( void* ) &m_clGridDimension );
	m_err |= clSetKernelArg( kernelDrawSample, 8, sizeof( cl_float ) * m_shOrder, NULL );
	m_err |= clSetKernelArg( kernelDrawSample, 9, sizeof( cl_float ) * datasetRaw->getNonZeroGradientIndexes().size(), NULL );
	m_err |= clSetKernelArg( kernelDrawSample, 10, sizeof( cl_mem ), ( void* ) &m_clSampleArray );
	m_err |= clSetKernelArg( kernelDrawSample, 11, sizeof( cl_mem ), ( void* ) &m_clTest );

	size_t globalSample[2] = { global, datasetRaw->getNonZeroGradientIndexes().size() };
	size_t localSample[2] = { 1, datasetRaw->getNonZeroGradientIndexes().size() };
	m_err |= clEnqueueNDRangeKernel( m_command_queue, kernelDrawSample, 2, NULL, globalSample, localSample, 0, NULL, &event );
	wlog::debug( "GPU" ) << "Samples drawn with " << global * datasetRaw->getNonZeroGradientIndexes().size() << " Threads! Errorcode: " << m_err;
	
	//m_err = clWaitForEvents( 1, &event );
	//m_err = clReleaseEvent( event );


	// compute odf
	m_err = 0;
	cl_kernel kernelOdf = clCreateKernel( m_program, "computeOdf", &m_err );
	m_err |= clSetKernelArg( kernelOdf, 0, sizeof( cl_mem ), ( void* ) &m_clOdf );
	m_err |= clSetKernelArg( kernelOdf, 1, sizeof( cl_mem ), ( void* ) &m_clOdfArray );
	m_err |= clSetKernelArg( kernelOdf, 2, sizeof( cl_mem ), ( void* ) &m_clSampleArray );
	m_err |= clSetKernelArg( kernelOdf, 3, sizeof( cl_mem ), ( void* ) &m_clDimensions );

	size_t globalOdf[2] = { global, getPot( m_reconstructionPoints.size() ) };
	size_t localOdf[2] = { 1, 32 };
	m_err |= clEnqueueNDRangeKernel( m_command_queue, kernelOdf, 2, NULL, globalOdf, localOdf, 0, NULL, &event );
	wlog::debug( "GPU" ) << "Computing odf. Errorcode: " << m_err;

	//m_err = clWaitForEvents( 1, &event );
	//m_err = clReleaseEvent( event );

	m_err = 0;
	cl_kernel kernelGetMaxima = clCreateKernel( m_program, "getMaxima", &m_err );
	m_err |= clSetKernelArg( kernelGetMaxima, 0, sizeof( cl_mem ), ( void* ) &m_clOdfArray );
	m_err |= clSetKernelArg( kernelGetMaxima, 1, sizeof( cl_mem ), ( void* ) &m_clProperties );
	m_err |= clSetKernelArg( kernelGetMaxima, 2, sizeof( cl_mem ), ( void* ) &m_clNeighbors );
	m_err |= clSetKernelArg( kernelGetMaxima, 3, sizeof( cl_mem ), ( void* ) &m_clPeaksArray );
	m_err |= clSetKernelArg( kernelGetMaxima, 4, sizeof( cl_mem ), ( void* ) &m_clDimensions );

	size_t globalMaxima[2] = { global, getPot( m_reconstructionPoints.size() ) };
	size_t localMaxima[2] = { 1, 32 };
	m_err |= clEnqueueNDRangeKernel( m_command_queue, kernelGetMaxima, 2, NULL, globalMaxima, localMaxima, 0, NULL, NULL );
	wlog::debug( "GPU" ) << "Getting local maximas. Errorcode: " << m_err;


	// wird im moment irgendwie nicht gebraucht da die lokalen maximas schon gut aussehen!
	/*m_err = 0;
	cl_kernel kernelEvaluateMaxima = clCreateKernel( m_program, "evaluateMaxima", &m_err );
	m_err |= clSetKernelArg( kernelEvaluateMaxima, 0, sizeof( cl_mem ), ( void* ) &m_clOdfArray );
	m_err |= clSetKernelArg( kernelEvaluateMaxima, 1, sizeof( cl_mem ), ( void* ) &m_clPeaksArray );
	m_err |= clSetKernelArg( kernelEvaluateMaxima, 2, sizeof( cl_mem ), ( void* ) &m_clMainDir );
	m_err |= clSetKernelArg( kernelEvaluateMaxima, 3, sizeof( cl_mem ), ( void* ) &m_clDimensions );
	m_err |= clSetKernelArg( kernelEvaluateMaxima, 4, sizeof( cl_mem ), ( void* ) &m_clReconstruction );

	size_t globalEvaluate = global;
	m_err |= clEnqueueNDRangeKernel( m_command_queue, kernelEvaluateMaxima, 1, NULL, &globalEvaluate, NULL, 0, NULL, NULL );
	wlog::debug( "GPU" ) << "Evaluating local maximas. Errorcode: " << m_err;*/

	m_err = 0;
	cl_kernel kernelIntegrate = clCreateKernel( m_program, "integrate", &m_err );
	m_err |= clSetKernelArg( kernelIntegrate, 0, sizeof( cl_mem ), ( void* ) &m_clFibers );
	m_err |= clSetKernelArg( kernelIntegrate, 1, sizeof( cl_mem ), ( void* ) &m_clScalarsOut );
	m_err |= clSetKernelArg( kernelIntegrate, 2, sizeof( cl_mem ), ( void* ) &m_clScalars );
	m_err |= clSetKernelArg( kernelIntegrate, 3, sizeof( cl_mem ), ( void* ) &m_clProperties );
	m_err |= clSetKernelArg( kernelIntegrate, 4, sizeof( cl_mem ), ( void* ) &m_clOdfArray );
	m_err |= clSetKernelArg( kernelIntegrate, 5, sizeof( cl_mem ), ( void* ) &m_clPeaksArray );
	m_err |= clSetKernelArg( kernelIntegrate, 6, sizeof( cl_mem ), ( void* ) &m_clReconstruction );
	m_err |= clSetKernelArg( kernelIntegrate, 7, sizeof( cl_mem ), ( void* ) &m_clDimensions );
	m_err |= clSetKernelArg( kernelIntegrate, 8, sizeof( cl_mem ), ( void* ) &m_clOldDirection );

	size_t globalIntegrate = global;
	size_t localIntegrate = local;
	m_err |= clEnqueueNDRangeKernel( m_command_queue, kernelIntegrate, 1, NULL, &globalIntegrate, &localIntegrate, 0, NULL, NULL );
	wlog::debug( "GPU" ) << "Integrated linesegment. Errorcode: " << m_err;

	// read back results
	m_err |= clEnqueueReadBuffer( m_command_queue, m_clOdfArray, CL_TRUE, 0, sizeof( float ) * m_reconstructionPoints.size() * global, odfArray, 0, NULL, NULL );
	m_err |= clEnqueueReadBuffer( m_command_queue, m_clTest, CL_TRUE, 0, sizeof( float ) * global, test, 0, NULL, NULL );
	m_err |= clEnqueueReadBuffer( m_command_queue, m_clPeaksArray, CL_TRUE, 0, sizeof( short ) * m_reconstructionPoints.size() * global, peaksArray, 0, NULL, NULL );
	m_err |= clEnqueueReadBuffer( m_command_queue, m_clMainDir, CL_TRUE, 0, sizeof( float ) * 4 * 12 * global, mainDir, 0, NULL, NULL );
	m_err |= clEnqueueReadBuffer( m_command_queue, m_clFibers, CL_TRUE, 0, sizeof( float ) * vertices * global, fiberOut, 0, NULL, NULL );
	m_err |= clEnqueueReadBuffer( m_command_queue, m_clScalarsOut, CL_TRUE, 0, sizeof( float ) * m_gridScalar->size(), scalarOut, 0, NULL, NULL );

	// clean up
	clReleaseKernel( kernelDrawSample );
	clReleaseKernel( kernelOdf );
	clReleaseKernel( kernelGetMaxima );
	clReleaseKernel( kernelIntegrate );

	// 	-------------------	TESTING  --------------------
	for( int i=0; i<m_reconstructionPoints.size(); i++ ) {
	//	wlog::debug( "GPU" ) << test[i];
		wlog::debug( "GPU" ) << peaksArray[ i * global + 0 ];
	}

	wlog::debug( "GPU" ) << "VoxelIndex: " << m_grid->getVoxelNum( startPosition );

	// testing spherical harmonic
	std::vector< WPosition > verts;
	WValue< double > odf = WValue< double >( m_reconstructionPoints.size() );
	for( int i=0; i<m_reconstructionPoints.size(); i++ ) {
		//wlog::debug( "GPU" ) << "GPU: " << odfArray[ i * global + 0 ];
		//odf[i] = odfArray[ i * global + 0 ];
		if( peaksArray[ i * global + 0 ] == -1 ) odf[i] = 0;
		else odf[i] = odfArray[ peaksArray[ i * global + 0 ] * global + 0 ];
	}
	for( int i=0; i<m_reconstructionPoints.size(); i++ ) {
		verts.clear();
		verts.push_back( startPosition );
		verts.push_back( startPosition + normalize( m_reconstructionPoints.at( i ) ) * odf[i] );
		m_fiberAcc.add( WFiber( verts ).asVector() );
	}

	boost::shared_ptr< WValueSet< double > > values = boost::shared_ptr< WValueSet< double > >( new WValueSet< double >( 0, 1, m_scalarData, W_DT_DOUBLE ) );
	boost::shared_ptr< WDataSetScalar > scalars( new WDataSetScalar( values, m_grid ) );

	boost::shared_ptr< WDataSetFibers > fibers = m_fiberAcc.buildDataSet();

	//releaseMemory();
	delete fiberOut;
	delete scalarOut;
	delete positionArray;
	delete sampleArray;
	delete odfArray;
	return std::pair< boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > >( fibers, scalars );
}

void WQBallGPU::createDataArrays() {

	b_Odf = new float[ m_odf.getNbCols() * m_odf.getNbRows() ];
	b_Base = new float[ m_baseMatrix.getNbCols() * m_baseMatrix.getNbRows() ];
	b_Coeff = new float[ m_coeffBase.getNbCols() * m_coeffBase.getNbRows() ];
	b_H = new float[ m_h.getNbCols() * m_h.getNbRows() ];
	b_Reconstruction = new cl_float4[ m_reconstructionPoints.size() ];
	b_Nzg = new ushort[ m_nonZeroGradients.size() ];
	b_Neighbors = new ushort[ m_neighbors.size() * 6 ];
	b_OldDirection = new cl_float4[ global ];

	b_Transformation = new short[ m_grid->getTransformationMatrix().getNbRows() * m_grid->getTransformationMatrix().getNbCols() ];

	b_Properties = new float[ 6 ];
	b_Dimensions = new uint[ 5 ];

	b_Properties[0] = static_cast< float >( m_stepSize->get( true ) );
	b_Properties[1] = static_cast< float >( m_stepDistance->get( true ) );
	b_Properties[2] = static_cast< float >( m_fa->get( true ) );
	b_Properties[3] = static_cast< float >( m_curv->get( true ) );
	b_Properties[4] = static_cast< float >( m_percentRange->get( true ) );
	b_Properties[5] = static_cast< float >( m_degreeRange->get( true ) );

	b_GridDimension.s[0] = m_grid->getNbCoordsX();
	b_GridDimension.s[1] = m_grid->getNbCoordsY();
	b_GridDimension.s[2] = m_grid->getNbCoordsZ();
	b_GridDimension.s[3] = 1;

	b_Dimensions[0] = datasetRaw->getNonZeroGradientIndexes().size();
	b_Dimensions[1] = m_shOrder;
	b_Dimensions[2] = m_reconstructionPoints.size();
	b_Dimensions[3] = m_grid->size();
	b_Dimensions[4] = 12;

	// odf
	for( int i=0; i<m_odf.getNbRows(); i++ ) {
		for( int j=0; j<m_odf.getNbCols(); j++ ) {
			b_Odf[ i * m_odf.getNbCols() + j ] = static_cast< float >( m_odf( i, j ) );
		}
	}

	// base
	for( int i=0; i<m_baseMatrix.getNbRows(); i++ ) {
		for( int j=0; j<m_baseMatrix.getNbCols(); j++ ) {
			b_Base[ i * m_baseMatrix.getNbCols() + j ] = static_cast< float >( m_baseMatrix( i, j ) );
		}
	}

	// coeff
	for( int i=0; i<m_coeffBase.getNbRows(); i++ ) {
		for( int j=0; j<m_coeffBase.getNbCols(); j++ ) {
			b_Coeff[ i * m_coeffBase.getNbCols() + j ] = static_cast< float >( m_coeffBase( i, j ) );
		}
	}

	// H
	for( int i=0; i<m_h.getNbRows(); i++ ) {
		for( int j=0; j<m_h.getNbCols(); j++ ) {
			b_H[ i * m_h.getNbCols() + j ] = static_cast< float >( m_h( i, j ) );
		}
	}

	// reconstruction
	for( int i=0; i<m_reconstructionPoints.size(); i++ ) {
		b_Reconstruction[i].s[0] = static_cast< float >( m_reconstructionPoints.at( i )[0] );
		b_Reconstruction[i].s[1] = static_cast< float >( m_reconstructionPoints.at( i )[1] );
		b_Reconstruction[i].s[2] = static_cast< float >( m_reconstructionPoints.at( i )[2] );
		b_Reconstruction[i].s[3] = 1.f;
	}

	// non zero gradients
	for( int i=0; i<m_nonZeroGradients.size(); i++ ) {
		b_Nzg[i] = static_cast< ushort >( m_nonZeroGradients.at( i ) );
	}

	// transformation
	WMatrix< double > transformation = m_grid->getTransformationMatrix();
	Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix( transformation.getNbRows(), transformation.getNbCols() );
	for( int row = 0; row < matrix.rows(); ++row ) {
        for( int col = 0; col < matrix.cols(); ++col ) {
            matrix( row, col ) = static_cast< double >( transformation( row, col ) );
        }
    }
    WMatrix< double > inverse = WMatrix< double >( (Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >)matrix.inverse() );

	for( int i=0; i<transformation.getNbRows(); i++ ) {
		for( int j=0; j<transformation.getNbCols(); j++ ) {
			b_Transformation[ i * transformation.getNbCols() + j ] = transformation( i, j );
		}
	}

	// neighbors
	for( int i=0; i<m_neighbors.size() * 6; i++ ) {
		b_Neighbors[i] = -1;
	}
	for( int i=0; i<m_neighbors.size(); i++ ) {
		for( int j=0; j<m_neighbors.at(i).size(); j++ ) {
			b_Neighbors[ j * m_neighbors.size() + i ] = m_neighbors.at(i)[j];
		}
	}

	//old directions
	for( int i=0; i<global; i++ ) {
		b_OldDirection[i].s[0] = 0.f;
		b_OldDirection[i].s[1] = 0.f;
		b_OldDirection[i].s[2] = 0.f;
		b_OldDirection[i].s[3] = 1.f;
	}
}

void WQBallGPU::releaseMemory() {
	clReleaseMemObject( m_clHardi );
	clReleaseMemObject( m_clScalars );
	clReleaseMemObject( m_clFibers );
	clReleaseMemObject( m_clScalarsOut );
	clReleaseMemObject( m_clOdf );
	clReleaseMemObject( m_clBase );
	clReleaseMemObject( m_clCoeff );
	clReleaseMemObject( m_clH );
	clReleaseMemObject( m_clReconstruction );
	clReleaseMemObject( m_clProperties );
	clReleaseMemObject( m_clNzg );
	clReleaseMemObject( m_clGridDimension );
	clReleaseMemObject( m_clDimensions );
	clReleaseMemObject( m_clStartPos );
	clReleaseMemObject( m_clPositionArray );
	clReleaseMemObject( m_clSampleArray );
	clReleaseMemObject( m_clOdfArray );
	clReleaseMemObject( m_clPeaksArray );
	clReleaseMemObject( m_clNeighbors );
	clReleaseMemObject( m_clMainDir );
	clReleaseMemObject( m_clOldDirection );
	clReleaseMemObject( m_clTest );
}

void WQBallGPU::cleanUp() {
	clReleaseProgram( m_program );
	clReleaseCommandQueue( m_command_queue );
	clReleaseContext( m_context );
	delete m_hardiData;
	delete m_scalarDataIn;
	delete b_Odf;
	delete b_Base;
	delete b_Coeff;
	delete b_H;
	delete b_Reconstruction;
	delete b_Properties;
	delete b_Nzg;
	delete b_Dimensions;
	delete b_Transformation;
	delete b_Neighbors;
	delete b_OldDirection;
}