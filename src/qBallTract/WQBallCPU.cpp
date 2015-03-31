#include <utility>

#include <omp.h>
#include <math.h>

// OW core headers
#include "core/kernel/WKernel.h"

#include "WQBallCPU.h"
#include "WQBallAlgorithm.h"

WQBallCPU::WQBallCPU():WObjectNDIP< WQBallAlgorithm >( "CPU", "QBall Tractography on CPU") {
	m_parallel = false;
	m_ready = false;


	// properies
	m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

	m_enableSH = m_properties->addProperty( "Show SH", "Show Spherical Harmonic at position", false );
	m_mainDir = m_properties->addProperty( "Main directions", "Main directions of corresponding SH", false );

	// cores
	m_cores = m_properties->addProperty( "Cores", "How many CPU cores should be used?", 1, m_propCondition );
	m_cores->setMin( 1 );
	m_cores->setMax( omp_get_max_threads() );

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

WQBallCPU::WQBallCPU( bool parallel ):WObjectNDIP< WQBallAlgorithm >( "CPU+", "QBall Tractography on CPU") {
	m_parallel = true;
	m_ready = false;


	// properies
	m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

	// cores
	m_cores = m_properties->addProperty( "Cores", "How many CPU cores should be used?", 1, m_propCondition );
	m_cores->setMin( 1 );
	m_cores->setMax( omp_get_max_threads() );
	m_cores->setHidden( true );

	m_enableSH = m_properties->addProperty( "Show SH", "Show Spherical Harmonic at position", false );
	m_mainDir = m_properties->addProperty( "Main directions", "Main directions of corresponding SH", false );
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
    m_degreeRange->setMin( 0 );
    m_degreeRange->setMax( 90 );

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

WQBallCPU::~WQBallCPU() {

}

void WQBallCPU::prepareAlgorithm( boost::shared_ptr< WDataSetRawHARDI > raw, boost::shared_ptr< WDataSetScalar > scalar, boost::filesystem::path path, boost::shared_ptr< WProgress > prog ) {
	WQBallAlgorithm::prepareAlgorithm( raw, scalar, path, prog );
}

std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > WQBallCPU::operator()( boost::shared_ptr< WProgress > progress, WPosition startPosition ) {
	//WQBallAlgorithm::operator()( progress, startPosition );
	m_ready = false;
	
	WRealtimeTimer timer = WRealtimeTimer();
    timer.reset();

   	fa = m_fa->get( true );
	curv = m_curv->get( true );
	stepSize = m_stepSize->get( true );
	stepDistance = m_stepDistance->get( true );
	percentRange = m_percentRange->get( true );
	degreeRange = m_degreeRange->get( true );
	int cores = m_cores->get( true );

	if( !m_enableSH->get() ) {

		wlog::debug( "CPU" ) << m_seedPoints->get( true );
		std::srand( std::time( NULL ) );
		if( m_parallel ) {
			if( m_parallel ) wlog::debug( "CPU+" ) << "parallel execution";
			omp_set_num_threads( 6 );
			#pragma omp parallel for schedule( dynamic )
			for(size_t i=0; i<m_seedPoints->get( true ); i++) {

				m_fiberAcc.add( algorithm( startPosition )->asVector() );

				++*progress;
			}
		} else {
			wlog::debug( "CPU" ) << "Computing on " << cores << " cores ";
			omp_set_num_threads( cores );
			#pragma omp parallel for schedule( dynamic )
			for(size_t i=0; i<m_seedPoints->get( true ); i++) {

				m_fiberAcc.add( algorithm( startPosition )->asVector() );

				++*progress;
			}
		}
		
	}
	progress->finish();
	// TODO vllt weg
	WValue< double > sample = WValue< double >( m_nonZeroGradients.size() );
	if( m_mainDir->get() && m_enableSH->get() ) {
		drawSample( startPosition, sample );
		WValue< double > sh = m_odf * sample;
		std::vector< int > peaks;
		getPeaks( sh, peaks );
		std::vector< WPosition > vertices;

		for( size_t i=0; i<peaks.size(); i++) {
		    vertices.clear();
		    vertices.push_back( startPosition );
			vertices.push_back( startPosition + ( normalize( m_reconstructionPoints.at( peaks.at( i ) ) ) * sh[ peaks.at( i ) ] ) );
			wlog::debug( "CPU" ) << sh[ peaks.at( i ) ];
		    m_fiberAcc.add( WFiber( vertices ).asVector() );
	    }
	    if( peaks.size() <= 1 ) {
	    	vertices.clear();
	    	vertices.push_back( startPosition );
	    	vertices.push_back( startPosition );
	    	m_fiberAcc.add( WFiber( vertices ).asVector() );
	    }
	}
	if( m_enableSH->get() && !m_mainDir->get() ) {
		drawSample( startPosition, sample );
		WValue< double > sh = m_odf * sample;
		std::vector< WPosition > vertices;
		for( size_t i=0; i<sh.size(); i++) {
		   	vertices.clear();
		   	vertices.push_back( startPosition );
		   	vertices.push_back( startPosition + ( normalize( m_reconstructionPoints.at( i ) ) * ( sh[i] / 1 ) ) );
		   	m_fiberAcc.add( WFiber( vertices ).asVector() );
		}
	}


	// scale scalar data
	for( int i=0; i<m_grid->size(); i++ ) {
		if( m_scalarData->at( i ) != 0.0 ) {
			m_scalarData->at( i ) = log( 1.0 + m_scalarData->at( i ) )  / log( m_seedPoints->get( true ) + 1 );
		}
	}
	boost::shared_ptr< WValueSet< double > > values = boost::shared_ptr< WValueSet< double > >( new WValueSet< double >( 0, 1, m_scalarData, W_DT_DOUBLE ) );
	boost::shared_ptr< WDataSetScalar > scalars( new WDataSetScalar( values, m_grid ) );

	boost::shared_ptr< WDataSetFibers > fibers = m_fiberAcc.buildDataSet();

	wlog::debug( "CPU" ) << "Time: " << timer.elapsed();
	m_ready = true;
	return std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > >( fibers, scalars );
}

boost::shared_ptr< WFiber > WQBallCPU::algorithm( WPosition lastPoint ) {
	std::vector< WPosition > vertices;
	vertices.reserve( stepDistance / stepSize ); // math: max capacity => reserve? 
	WPosition newDirection = WPosition(0, 0, 0);	// TEST
	WPosition oldDirection = lastPoint;

	std::vector< int > peaks; // math: evtl boost array, oder auch reserve?

	WValue< double > sample( m_neighbors.size() );
	WValue< double > retSample( m_nonZeroGradients.size() );

	int voxelIndex = -1;
	bool breakCondition = false;
	float fAngle;

	while( m_grid->encloses( lastPoint + newDirection ) && !breakCondition ) {

		// scalar data
		if( voxelIndex != m_grid->getVoxelNum( lastPoint ) ) {
			voxelIndex = m_grid->getVoxelNum( lastPoint );
			m_scalarData->at( voxelIndex ) += 1.0;
		}

		// fiber data
		lastPoint += newDirection;
		vertices.push_back( lastPoint );


		drawSample( lastPoint, retSample );
		sample = m_odf * retSample;

		getPeaks( sample, peaks );

		oldDirection = newDirection;

		if( peaks.size() > 1 ) {
			std::sort( peaks.begin(), peaks.begin() + peaks.size() );
			newDirection = rndDist( rndGen ) % 2 == 0 ? m_reconstructionPoints.at( peaks.at( 0 ) ) : m_reconstructionPoints.at( peaks.at( 1 ) );
		} else newDirection = m_reconstructionPoints.at( peaks.at( 0 ) );

		// break conditions:
		// 		- curvature
		fAngle = angle( oldDirection, newDirection );
		if( abs( fAngle ) > curv ) {
			if( peaks.size() > 1 ) {
				/*while( peaks.size() != 0 ) {
					int rnd = rndDist( rndGen ) % peaks.size();
					newDirection = m_reconstructionPoints.at( peaks.at( rnd ) );
					fAngle = angle( oldDirection, newDirection );
					if( abs( fAngle ) < curv ) break;
					else peaks.erase( peaks.begin() + rnd );
				}*/
				for( int i=0; i<peaks.size(); i++ ) {
					newDirection =  m_reconstructionPoints.at( peaks.at( i ) );
					fAngle = angle( oldDirection, newDirection );
					if( abs( fAngle ) < curv ) break;
				}
			}
			if( abs( fAngle ) > curv ) breakCondition = true;
		}

		//		- distance
		if( !breakCondition && ( vertices.size() - 1 ) * stepSize > stepDistance ) breakCondition = true;

		// 		- fractional anisotropy
		WVector3i vCoords = m_gridScalar->getVoxelCoord( lastPoint + newDirection );
		if( !breakCondition && datasetScalar->getValueAt( vCoords[0], vCoords[1], vCoords[2] ) < fa ) breakCondition = true;

		newDirection = normalize( newDirection ) * stepSize;
	}

	m_clearCache = true;
	boost::shared_ptr< WFiber > fiber( new WFiber( vertices ) );
	return fiber;
}

void WQBallCPU::drawSample( WPosition &position, WValue< double > &retValue ) {
	size_t index = m_grid->getVoxelNum( position );

    // get non zero gradient signals (F in berman paper) 
	WValue< double > nonZeroGradientSignals( datasetRaw->getNonZeroGradientIndexes().size() );
	for( size_t i=0; i<m_nonZeroGradients.size(); i++ ) {
		nonZeroGradientSignals[i] = datasetRaw->getValueSet()->getWValueDouble( index )[ m_nonZeroGradients.at( i ) ];
	}

	// calculate coefficients s
	WValue< double > s = m_coeffBase * nonZeroGradientSignals;

	// residuals			   =    fitted values     - 		f
	nonZeroGradientSignals = ( m_baseMatrix * s ) - nonZeroGradientSignals;

    // adjustment of residuals similar to Berman paper
    for(size_t i=0; i<nonZeroGradientSignals.size(); i++) {
    	nonZeroGradientSignals[i] /= std::sqrt( 1 - m_h(i, i) );
    }

    double scalar = 0.0;
    for( size_t i=0; i<retValue.size(); i++ ) {
    	scalar = 0.0;
    	for(size_t j=0; j<m_baseMatrix.getNbCols(); j++) {
    		scalar += m_baseMatrix(i, j) * s[j];
    	}
    	retValue[i] = scalar + nonZeroGradientSignals[ (rand() % nonZeroGradientSignals.size()) ];
    } 
}

void WQBallCPU::getPeaks( WValue< double > &sample, std::vector< int > &peaks ) {
	peaks.clear();
	peaks.reserve( 10 );
	peaks.push_back( 0 );

	float max = 0.f;
	size_t maxIndex = 0;
	for( size_t i=0; i<sample.size(); i++ ) {
		// global maximum
		if( abs( sample[i] ) > max ) {
			max = abs( sample[i] );
			maxIndex = i;
		}
	}
	peaks[0] = maxIndex;

	for( size_t i=0; i<sample.size(); i++ ) {
		if( i != maxIndex ) {
			// local maxima
			bool isLocalMax = true;
			for( size_t j=0; j<m_neighbors.at( i ).size(); j++ ) {
				if( abs( sample[ i ] ) < abs( sample[ m_neighbors.at( i )[ j ] ] ) ) isLocalMax = false;
			}
			//		check 33% range
			//if( isLocalMax && abs( sample[ i ] ) < abs( sample[ maxIndex ] ) * ( 1.0 - percentRange ) ) isLocalMax = false;
			//		check angle
			if( isLocalMax ) {
				for( size_t j=0; j<peaks.size(); j++ ) {
					if( i != peaks.at( j ) && abs( angle( m_reconstructionPoints.at( i ), m_reconstructionPoints.at( peaks.at( j ) ) ) ) < degreeRange ) isLocalMax = false;
					if( !isLocalMax ) break;
				}
			}
			if( isLocalMax ) peaks.push_back( i );
		}
	}
}

int WQBallCPU::getSeedSize() { return m_seedPoints->get( true ); }