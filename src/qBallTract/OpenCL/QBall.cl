#define STEP_SIZE = 0;
#define DISTANCE = 1;
#define FA = 2;
#define CURV = 3;
#define RANGE = 4;
#define DEGREE = 5;

#define GRADIENTS = 0;
#define SH_ORDER = 1;
#define REC_POINTS = 2;
#define GRID_SIZE = 3;
#define DIRECTIONS = 4;

float4 getVoxelNum( float4 pos, short *transformation, ushort4 *gridDimension ) {

}

__kernel void algorithm(__global const short *hardi, __global const float *scalar,
	__global float *outputFibers			, __global float *outputScalar,
	__global const float *odf				, __global const float *base, 
	__global const float *coeff				, __global const float *h, 
	__global const float4 *reconstruction	, __global const float *properties, 
	__global const ushort *nzg				, __global const ushort4 *gridDimension, 
	__global const uint *dimensions		, __global const float4 *startPos) 
{


	int id = get_global_id(0);
									// !!! HARDCODE
	outputScalar[id] = (float)hardi[ 67 * id + 9 ];
	//outputFibers[id] = dimensions[0];
}

/*
	global = { 512, 60 }
	local = { 1, 60 }
*/
__kernel void drawSample( __global const short *hardi, __global const ushort *nzg,
	__global const float *base				, __global const float *coeff,
	__global const float *h 				, __global const uint *dimensions,
	__global int4 *positionArray			, __global const ushort4 *gridDimension,
	__local float *coefficients				, __local float *residuals,
	__global float *sampleArray				, __global float *output ) 
{

	// get voxelCoord
	int4 startPos = positionArray[ get_global_id(0) ];
	int voxelIndex = startPos.x + startPos.y * gridDimension->x + startPos.z * gridDimension->x * gridDimension->y;
	output[ get_global_id(0) ] = voxelIndex;

	// computing coefficients
	//TODO nicht ganz optimal, da 45 cores nichts tun
	if( get_global_id(1) < dimensions[1] ) {
		float s = 0;
		for( int i=0; i<dimensions[0]; i++ ) {
			s += coeff[ get_global_id(1) * dimensions[0] + i ] * hardi[ voxelIndex * 67 + nzg[i] ];
		}
		coefficients[ get_global_id(1) ] = s;
	}

	barrier( CLK_LOCAL_MEM_FENCE );

	// residuals
	float residual = 0.f;
	for( int i=0; i<dimensions[1]; i++ ) {
		residual += base[ get_global_id(1) * dimensions[1] + i ] * coefficients[ i ];
	}
	residual -= hardi[ voxelIndex * 67 + nzg[ get_global_id(1) ] ];

	// adjustment
	float scalar = 0.f;
	residual /= sqrt( 1 - h[ get_global_id(1) * dimensions[0] + get_global_id(1) ] );
	for( int i=0; i<dimensions[1]; i++ ) {
		scalar += base[ get_global_id(1) * dimensions[1] + i ] * coefficients[ i ];
	}
	residuals[ get_global_id(1) ] = residual;
																			// residuals[ random num ]!!
	sampleArray[ get_global_id(1) * get_global_size(0) + get_global_id(0) ] = scalar + residual;
	//if( get_global_id(1) == 1 ) output[get_global_id(0)] = scalar + residual;
	
}

/*
	global = { 512, 1024 }
	local = { 1, 32 }
*/
__kernel void computeOdf( __global const float *odf, __global float *odfArray,
 	__global float *sampleArray				, __global const uint *dimensions ) 
{
	
	if( get_global_id(1) >= dimensions[2] ) return;

	float value = 0.f;
	for( int i=0; i<dimensions[0]; i++ ) {							// evt. dim - i da reihenfolge anders gespeichert
		value += odf[ get_global_id(1) * dimensions[0] + i ] * sampleArray[ i * get_global_size(0) + get_global_id(0) ];
	}

	odfArray[ get_global_id(1) * get_global_size(0) + get_global_id(0) ] = value;
	


}

/*
	global = { 512, 1024 }
	local = { 1, 32 }
*/
__kernel void getMaxima( __global float *odfArray, __global const float *properties,
	__global const ushort *neighbors			, __global short *peaksArray, 
	__global const uint *dimensions ) 
{

	if( get_global_id(1) >= dimensions[2] ) return;


	float odfValue = odfArray[ get_global_id(1) * get_global_size(0) + get_global_id(0) ];
	odfValue = odfValue < 0 ? -odfValue : odfValue;

	ushort n[6];
	for( int i=0; i<6; i++ ) {
		n[i] = neighbors[ i * dimensions[2] + get_global_id(1) ];
	}

	peaksArray[ get_global_id(1) * get_global_size(0) + get_global_id(0) ] = -1;
	float temp;
	for( int i=0; i<6; i++ ) {
		if( n[i] == -1 ) continue;
		temp = odfArray[ n[i] * get_global_size(0) + get_global_id(0) ];
		temp = temp < 0 ? -temp : temp;
		if( odfValue < temp ) return;
	}

	peaksArray[ get_global_id(1) * get_global_size(0) + get_global_id(0) ] = get_global_id(1);
	//if( isLocalMax == 0 ) peaksArray[ get_global_id(1) * get_global_size(0) + get_global_id(0) ] = -1;

}

/*
	global = 512
*/
__kernel void evaluateMaxima( __global float *odfArray, __global short *peaksArray,
	__global float4 *mainDir					, __global const uint *dimensions,
	__global float4 *reconstruction )
{
	// sortieren (bubblesort)
	/*int n=0;
	int maxIndex;
	float max, v;
	while( true ) {
		bool swapped = true;
		max = 0.f;
		for( int i=n; i<dimensions[2]; i++ ) {
			v = odfArray[ i * get_global_size(0) + get_global_id(0) ];
			v = v < 0 ? -v : v;
			if( v > max ) {
				max = v;
				maxIndex = i;
			}
		}
		if( max != -1 ) {
			// swap in odfArray
			float temp = odfArray[ maxIndex * get_global_size(0) + get_global_id(0) ];
			odfArray[ maxIndex * get_global_size(0) + get_global_id(0) ] = odfArray[ n * get_global_size(0) + get_global_id(0) ];
			odfArray[ n * get_global_size(0) + get_global_id(0) ] = temp;

			// swap in peaksArray
			short tempPeaks = peaksArray[ maxIndex * get_global_size(0) + get_global_id(0) ];
			peaksArray[ maxIndex * get_global_size(0) + get_global_id(0) ] = peaksArray[ n * get_global_size(0) + get_global_id(0) ];
			peaksArray[ n * get_global_size(0) + get_global_id(0) ] = tempPeaks;

			n++;
		} else swapped = false;
		if( !swapped ) break;
	}

	mainDir[0 * get_global_size(0) + get_global_id(0)] = reconstruction[ peaksArray[0] ];
	mainDir[1] = reconstruction[ peaksArray[0] ] * -1;*/
}

__kernel void integrate( __global float *outputFibers, __global float *outputScalar,
	__global const float *scalar 					, __global float *properties,
	__global float *odfArray						, __global short *peaksArray,
	__global const float4 *reconstruction			, __global const uint *dimensions,
	__global float4 *oldDirections )
{
	short mainDirections[12];
	float directionValues[12];
	for( int i=0; i<12; i++ ) {
		mainDirections[i] = -1;
		directionValues[i] = -1;
	}

	//fill direction array
	int count = 0;
	for( int i=0; i<dimensions[2]; i++ ) {
		if( peaksArray[ i * get_global_size(0) + get_global_id(0) ] != -1 ) {
			mainDirections[ count ] = peaksArray[ i * get_global_size(0) + get_global_id(0) ];
			directionValues[ count ] = odfArray[ i * get_global_size(0) + get_global_id(0) ];
			count++;
		}
	}

	// sort arrays
	bool swapped = true;
	int n = count;
	while( swapped ) {
		for( int i=n; i>0; i-- ) {
			if( directionValues[i] > directionValues[i-1] ) {
				float temp = directionValues[i];
				directionValues[i] = directionValues[i-1];
				directionValues[i-1] = temp;

				short tempDir = mainDirections[i];
				mainDirections[i] = mainDirections[i-1];
				mainDirections[i-1] = tempDir;

				swapped = true;
			}
		}
		n--;
	}

	float4 direction;
	// if new fiber
	if( oldDirections[ get_global_id(0) ].x == 0 && oldDirections[ get_global_id(0) ].y == 0 && oldDirections[ get_global_id(0) ].z == 0 ) {
		direction = get_global_id(0) % 2 == 0 ? mainDirections[0] : mainDirections[1];
	} else {
		direction = mainDirections[0];
	}



}

float angle( float4 v1, float4 v2 ) {
	float f = dot( v1, v2 ) / ( length( v1 ) * length( v2 ) );
	if( f > 1.0 ) f = 1.0;
	if( f < -1.0 ) f = -1.0;
	return acos( f ) * 180 / M_PI;
}