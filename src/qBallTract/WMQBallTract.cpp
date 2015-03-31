//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <string>
#include <math.h>

#include <osg/Geode>
#include <osg/ShapeDrawable>

#include "core/dataHandler/WDataSetRawHARDI.h"
#include "core/dataHandler/WDataSetSphericalHarmonics.h"
#include "core/dataHandler/WDataSetScalar.h"
#include "core/dataHandler/WDataSetPoints.h"
#include "core/dataHandler/WDataSetFibers.h"
#include "core/dataHandler/WDataSetVector.h"
#include "core/dataHandler/WDataHandler.h"
#include "core/dataHandler/WFiberAccumulator.h"
#include "core/kernel/WKernel.h"
#include "core/kernel/WSelectionManager.h"
#include "core/common/math/WMath.h"
#include "core/common/WPropertyHelper.h"
#include "core/common/WStrategyHelper.h"
#include "core/graphicsEngine/WGEGeodeUtils.h"

//#include "WGeometryFunctions.h"
#include "WMQBallTract.h"

WMQBallTract::WMQBallTract():
	WModule(), m_strategy( "QBall Tracking",
    "Select one of the algorithms and configure it to your needs.", NULL, "Algorithm", "A list of algorithms" )
{	
	m_qBallCPU = WQBallCPU::SPtr( new WQBallCPU() );
	m_qBallCPU_parallel = WQBallCPU::SPtr( new WQBallCPU( true ) );
	m_qBallGPU = WQBallGPU::SPtr( new WQBallGPU() );

	m_strategy.addStrategy( m_qBallCPU );
	m_strategy.addStrategy( m_qBallCPU_parallel );
	m_strategy.addStrategy( m_qBallGPU );
}

WMQBallTract::~WMQBallTract() {
}

boost::shared_ptr< WModule > WMQBallTract::factory() const {
	return boost::shared_ptr< WModule >( new WMQBallTract() );
}

const char** WMQBallTract::getXPMIcon() const {
	return NULL;
}

const std::string WMQBallTract::getName() const {
	return "QBall Tractography";
}

const std::string WMQBallTract::getDescription() const {
	return "TO DO!";
}

void WMQBallTract::connectors() {

	m_inputRaw = WModuleInputData< WDataSetRawHARDI >::createAndAdd( shared_from_this(), "HARDI", "HARDI dataset" );
	m_inputScalar = WModuleInputData< WDataSetScalar >::createAndAdd( shared_from_this(), "fa", "fractional anisotropy" );

	m_output = WModuleOutputData< WDataSetFibers >::createAndAdd( shared_from_this(), "fibers", "fibers to display" );
	m_outputScalar = WModuleOutputData< WDataSetScalar >::createAndAdd( shared_from_this(), "scalar data", "scalar data for visited voxels" );

	WModule::connectors();

}

void WMQBallTract::properties() {

	m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

	m_aTrigger = m_properties->addProperty( "Set marker", "Set marker to intersection of planes.", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_TriggerStartCalculate = m_properties->addProperty( "Start calculation", "Start calculation",
        WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_pushToCache = m_properties->addProperty( "Push to cache", "Save current scalar data.", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_clearCache = m_properties->addProperty( "Clear cache", "Delete current scalar data.", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );


    // Adding selection tools for the coordinates
    m_xValue = m_properties->addProperty( "Select x value", "x value", 100.0, m_propCondition );
    m_yValue = m_properties->addProperty( "Select y value", "y value", 100.0, m_propCondition );
    m_zValue = m_properties->addProperty( "Select z value", "z value", 100.0, m_propCondition );

    m_order = m_properties->addProperty( "SH order", "Spherical Harmonic order", 4, m_propCondition );
    m_order->setMin( 2 );
    m_order->setMax( 8 );
    m_reconstructions = m_properties->addProperty( "Reconstruction points", "Number of reconstruction points", 642, m_propCondition );
    m_reconstructions->setMin( 60 );
    m_tesselationLevel = m_properties->addProperty( "Tesselation level", "Order of icosahedron tesselation", 3, m_propCondition );
    m_tesselationLevel->setMin( 1 );
    m_tesselationLevel->setMax( 4 );

    m_properties->addProperty( m_strategy.getProperties() );

	WModule::properties();

}

void WMQBallTract::requirements() {
	// image extractor +  nav slices
}

void WMQBallTract::moduleMain() {

	m_moduleState.setResetable(true, true);
	m_moduleState.add( m_inputRaw->getDataChangedCondition() );

	m_moduleState.add( m_propCondition );

	std::string strategyName = "";

	ready();

	// Insert OSG node into the scene
    m_rootNode = new WGEManagedGroupNode( m_active );
    WKernel::getRunningKernel()->getGraphicsEngine()->getScene()->insert( m_rootNode );

	while( !m_shutdownFlag() ) {
		
		debugLog() << "Waiting ... ";
		m_moduleState.wait();

		if( m_shutdownFlag ) {
			break;
		}

		if( m_inputRaw->updated() || m_inputScalar->updated() ) {

			datasetRaw = m_inputRaw->getData();
			datasetScalar = m_inputScalar->getData();
			bool dataValid = ( datasetRaw ) && ( datasetScalar );

			if( dataValid ) {
				debugLog() << "non zero gradients: " << datasetRaw->getNonZeroGradientIndexes().size();
				debugLog() << "number of measurements: " << datasetRaw->getNumberOfMeasurements();
				debugLog() << "Size: " << datasetRaw->getValueSet()->size();
				debugLog() << "Dimension: " << datasetRaw->getValueSet()->dimension();
				debugLog() << "OrientationSize: " << datasetRaw->getOrientations().size();
				debugLog() << "Voxel size: " << datasetRaw->getValueSet()->getWValueDouble(0).size();

				// get 3d grid
				m_grid = boost::dynamic_pointer_cast< WGridRegular3D >( datasetRaw->getGrid() );
				m_gridScalar = boost::dynamic_pointer_cast< WGridRegular3D >( datasetScalar->getGrid() );
				prepareMatrices();
			} else {
				debugLog() << "No valid data. Cleaning up.";
				WKernel::getRunningKernel()->getGraphicsEngine()->getScene()->remove( m_rootNode );
			}
		}

		if( datasetRaw && datasetScalar && strategyName != m_strategy()->getName() ) {
			boost::shared_ptr< WProgress > progressPrepare( new WProgress( "preparing data" ) );
			m_progress->addSubProgress( progressPrepare );
			m_strategy()->prepareAlgorithm( datasetRaw, datasetScalar, getLocalPath(), progressPrepare );
			progressPrepare->finish();
			strategyName = m_strategy()->getName();			
		}

		if( m_xValue->changed() || m_yValue->changed() || m_zValue->changed() ) {

		}

		if( m_aTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) {
        	m_xValue->set( WKernel::getRunningKernel()->getSelectionManager()->getPropSagittalPos() );
            m_yValue->set( WKernel::getRunningKernel()->getSelectionManager()->getPropCoronalPos() );
            m_zValue->set( WKernel::getRunningKernel()->getSelectionManager()->getPropAxialPos() );

            // Reactive trigger
            m_aTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, false );
        }

        if( m_reconstructions->changed() || m_order->changed() || m_tesselationLevel->changed() ) {
        	prepareMatrices();
        }

        if( m_pushToCache->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) {
        	m_strategy()->setClearCache( false );
        	m_pushToCache->set( WPVBaseTypes::PV_TRIGGER_READY, false );
        }

        if( m_clearCache->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) {
        	m_strategy()->clearScalarData();
        	m_outputScalar->reset();
        	m_clearCache->set( WPVBaseTypes::PV_TRIGGER_READY, false );
        }

        if( m_TriggerStartCalculate->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED && m_strategy()->isReady() ) {

        	boost::shared_ptr< WProgress > progressCopy( new WProgress( "copying data" ) );
        	m_progress->addSubProgress( progressCopy );
        	m_strategy()->setOdf( m_odf );
        	m_strategy()->setBase( m_baseMatrix );
        	m_strategy()->setCoeffBase( m_coeffBase );
        	m_strategy()->setRegularization( m_h );
        	m_strategy()->setNeighbors( m_neighbors );
        	m_strategy()->setOrientations( m_reconstructionPoints );
        	m_strategy()->setSHOrder( m_shOrders.size() );
        	progressCopy->finish();

        	boost::shared_ptr< WProgress > progressTrack( new WProgress("tracking", m_strategy()->getSeedSize() ) );
        	m_progress->addSubProgress( progressTrack );
        	std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > results = m_strategy()->operator()( progressTrack, WPosition( m_xValue->get( true ), m_yValue->get( true ), m_zValue->get( true ) ) );
        	progressTrack->finish();

        	m_progress->removeSubProgress( progressTrack );
	        m_TriggerStartCalculate->set( WPVBaseTypes::PV_TRIGGER_READY, false );

			m_output->updateData( results.first );
			m_outputScalar->updateData( results.second );
        }
		debugLog() << "ready";

	}

	// clean up
    m_strategy()->cleanUp();
	WKernel::getRunningKernel()->getGraphicsEngine()->getScene()->remove( m_rootNode );

}

void WMQBallTract::prepareMatrices() {
	m_orientations.clear();
	m_reconstructionPoints.clear();

	// transform gradient directions from vectors to spherical polar coordinates
	std::vector< size_t > nonZeroGradients = datasetRaw->getNonZeroGradientIndexes();
	for(size_t i=0; i<nonZeroGradients.size(); i++) {
		m_orientations.push_back( 
			WUnitSphereCoordinates< double >( 
				WMatrixFixed< double, 3, 1 >( 
					datasetRaw->getOrientations().at(nonZeroGradients.at(i))[0],
					datasetRaw->getOrientations().at(nonZeroGradients.at(i))[1],
					datasetRaw->getOrientations().at(nonZeroGradients.at(i))[2]
				)
			)
		);
	}

	m_baseMatrix = WSymmetricSphericalHarmonic< double >().calcBaseMatrix( m_orientations, m_order->get( true ) );

	// transform reconstruction points to polar coordinates
	std::vector< WVector3d > spherePoints = sphereDistribution();
	for(size_t i=0; i<spherePoints.size(); i++) {
		m_reconstructionPoints.push_back(
			WUnitSphereCoordinates< double >(
				WMatrixFixed< double, 3, 1 >(
					spherePoints.at(i)[0],
					spherePoints.at(i)[1],
					spherePoints.at(i)[2]
				)
			)
		);
	}

	m_reconstructionMatrix = WSymmetricSphericalHarmonic< double >().calcBaseMatrix( m_reconstructionPoints, m_order->get( true ) );
	
	// create inverse of m_baseMatrix (base matrix)
	WMatrix< double > quadraticZQ = m_baseMatrix.transposed() * m_baseMatrix;
	// convert to eigenmatrix
	Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix( quadraticZQ.getNbRows(), quadraticZQ.getNbCols() );
	for( int row = 0; row < matrix.rows(); ++row ) {
        for( int col = 0; col < matrix.cols(); ++col ) {
            matrix( row, col ) = static_cast< double >( quadraticZQ( row, col ) );
        }
    }

    // compute inverse: (z^T * z)^-1
    m_inverse = WMatrix< double >( (Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >)matrix.inverse() );

    // zU * p * zQ+ * F
	// the pseudoinverse is computed without regularization! See Hess et al 2006 for further infomation
	calcLegendreMatrix();
	WMatrix< double > pInverse = m_inverse * m_baseMatrix.transposed();
	m_odf = m_reconstructionMatrix * ( m_p * pInverse );

	m_coeffBase = m_inverse * m_baseMatrix.transposed();
	m_h = m_baseMatrix * m_coeffBase;
}

void WMQBallTract::calcLegendreMatrix() {
	m_shOrders.clear();
	// reconstruct sh orders
	for( int i=0; i<=m_order->get(); i+=2 ) {
		for( int j=-i; j<=i; j++) {
			m_shOrders.push_back(i);
		}
	}
	debugLog() << "SH Order size: " << m_shOrders.size();

	m_p = WMatrix< double >( ( ( m_order->get() + 1) * ( m_order->get() + 2 ) ) / 2 );
	for(size_t i=0; i<m_shOrders.size(); i++) {
		int l = m_shOrders.at(i);
		if( l == 0 ) m_p(i, i) = 0;
		else {
			double enumerator = 1;
			double denominator = 1;
			for( size_t j=1; j<=l; j++ ) {
				if( j % 2 != 0 ) enumerator *= j;
				else denominator *= j;
			}
			double fraction = enumerator / denominator;
			//debugLog() << "Legendre: " << fraction;
			m_p(i, i) = std::pow(-1, l/2) * fraction;
		}
	}
}

std::vector< WVector3d > WMQBallTract::sphereDistribution() {
	std::vector< WVector3d > *vertices = new std::vector< WVector3d >;
	std::vector< unsigned int > *triangles = new std::vector< unsigned int >;

	tesselateIcosahedron( vertices, triangles, m_tesselationLevel->get( true ) );

	m_neighbors.clear();
	m_neighbors.resize( vertices->size() );
	for( size_t i=0; i<vertices->size(); i++) {
		for( size_t j=0; j<triangles->size(); j++ ) {
			if( i == triangles->at( j ) ) {
				int pos = j % 3;
				if( pos == 0 ) {
					m_neighbors[ i ].push_back( triangles->at( j + 1 ) );
					m_neighbors[ i ].push_back( triangles->at( j + 2 ) );
					j += 2;
				}
				if( pos == 1 ) {
					m_neighbors[ i ].push_back( triangles->at( j - 1 ) );
					m_neighbors[ i ].push_back( triangles->at( j + 1 ) );
					j += 1;
				}
				if( pos == 2 ) {
					m_neighbors[ i ].push_back( triangles->at( j - 1 ) );
					m_neighbors[ i ].push_back( triangles->at( j - 2 ) );
				}
				//check if last two added indices are already listed
				bool erased = false;
				int size = m_neighbors[ i ].size();
				for( size_t z=0; z<m_neighbors[ i ].size() - 1; z++ ) {
					if( m_neighbors[ i ].at( z ) == m_neighbors[ i ].at( size - 1 ) ) {
						m_neighbors[ i ].pop_back();
						erased = true;
						break;
					}
				}
				if( erased ) {
					size = m_neighbors[ i ].size();
					for( size_t z=0; z<m_neighbors[ i ].size() - 1; z++ ) {
						if( m_neighbors[ i ].at( z ) == m_neighbors[ i ].at( size - 1 ) ) {
							m_neighbors[ i ].pop_back();
							break;
						}
					}
				} else {
					for( size_t z=0; z<m_neighbors[ i ].size() - 2; z++ ) {
						if( m_neighbors[ i ].at( z ) == m_neighbors[ i ].at( size - 2 ) ) {
							m_neighbors[ i ].erase( m_neighbors[ i ].begin() + size - 2 );
							break;
						}
					}
				}
			}
		}
	}

	std::vector< WVector3d > returnVec = *vertices;
	delete vertices;
	delete triangles;
	return returnVec;
}

void WMQBallTract::activate() {

	if( m_active->get() ) {
		debugLog() << "Activate.";
	} else {
		debugLog() << "Deactivate.";
	}

	WModule::activate();

}













