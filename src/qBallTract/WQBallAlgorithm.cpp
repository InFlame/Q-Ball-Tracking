#include <utility>

#include "WQBallAlgorithm.h"

WQBallAlgorithm::~WQBallAlgorithm() {

}

void WQBallAlgorithm::prepareAlgorithm( boost::shared_ptr< WDataSetRawHARDI > raw, boost::shared_ptr< WDataSetScalar > scalar, boost::filesystem::path path, boost::shared_ptr< WProgress > prog ) {
	datasetRaw = raw;
	datasetScalar = scalar;

	m_grid = boost::dynamic_pointer_cast< WGridRegular3D >( datasetRaw->getGrid() );
	m_gridScalar = boost::dynamic_pointer_cast< WGridRegular3D >( datasetScalar->getGrid() );

	m_path = path;

	m_scalarData = boost::shared_ptr< std::vector< double > >( new std::vector< double > );										
	for( int i=0; i<raw->getGrid()->size(); i++ ) {
		m_scalarData->push_back( 0.0 );
	}
	//m_scalars = boost::shared_ptr< WValueSet< double > >( new WValueSet< double >( 0, 1, scalarData, W_DT_DOUBLE ) );

	m_nonZeroGradients = raw->getNonZeroGradientIndexes();

	// transformation TODO hardcoded
    m_transformation = m_transformation.identity();
    //m_transformation(0, 0) *= -1;
    m_transformation(1, 1) *= -1;
    m_transformation(2, 2) *= -1;	
}

std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > WQBallAlgorithm::operator()( boost::shared_ptr< WProgress > progress, WPosition startPosition ) {
	//if( m_clearCache ) clearScalarData();
}

bool WQBallAlgorithm::isReady() {
	return m_ready;
}

WValue< double > WQBallAlgorithm::drawSample( WPosition ) {

}

boost::shared_ptr< WFiber > WQBallAlgorithm::algorithm( WPosition ) {

}

void WQBallAlgorithm::cleanUp() {

}

int WQBallAlgorithm::getSeedSize() { 
	
}

void WQBallAlgorithm::clearScalarData() {
	m_scalarData = boost::shared_ptr< std::vector< double > >( new std::vector< double > );
	for( int i=0; i<datasetRaw->getGrid()->size(); i++ ) {
		m_scalarData->push_back( 0.0 );
	}
}

void WQBallAlgorithm::setOdf( WMatrix< double > odf ) { m_odf = odf; }
void WQBallAlgorithm::setBase( WMatrix< double > base ) { m_baseMatrix = base; }
void WQBallAlgorithm::setCoeffBase( WMatrix< double > cBase ) { m_coeffBase = cBase; }
void WQBallAlgorithm::setRegularization( WMatrix< double > h ) { m_h = h; }
void WQBallAlgorithm::setNeighbors( std::vector< std::vector < unsigned int > > neighbors ) { m_neighbors = neighbors; }
void WQBallAlgorithm::setOrientations( std::vector< WUnitSphereCoordinates< double > > reconstructionPoints ) { 
	m_reconstructionPoints.clear();
	for( size_t i=0; i< reconstructionPoints.size(); i++) {
		m_reconstructionPoints.push_back( m_transformation * reconstructionPoints.at( i ).getEuclidean() ); 
	}
}
void WQBallAlgorithm::setSHOrder( uint shOrder ) { m_shOrder = shOrder; }

void WQBallAlgorithm::setClearCache( bool c ) { m_clearCache = c; }
