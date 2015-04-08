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

#ifndef WMQBALLTRACT_H
#define WMQBALLTRACT_H

#include <string>

#include "core/kernel/WModule.h"
#include "core/kernel/WModuleInputData.h"
#include "core/kernel/WModuleOutputData.h"
#include "core/common/math/linearAlgebra/WMatrixFixed.h"
#include "core/common/WStrategyHelper.h"
#include "core/graphicsEngine/WGEManagedGroupNode.h"

#include "WQBallGPU.h"
#include "WQBallCPU.h"

class WMQBallTract: public WModule {

public:
	WMQBallTract();
	virtual ~WMQBallTract();

	virtual const std::string getName() const;
	virtual const std::string getDescription() const;
	virtual const char** getXPMIcon() const;

	virtual boost::shared_ptr< WModule > factory() const;

protected:
	virtual void moduleMain();
	virtual void connectors();
	virtual void properties();
	virtual void requirements();
	virtual void activate();

	osg::ref_ptr< WGEManagedGroupNode > m_rootNode;
    osg::ref_ptr< osg::Geode > m_geode;

private:

	/* connectors */
	boost::shared_ptr< WModuleInputData< WDataSetRawHARDI > > m_inputRaw;
	boost::shared_ptr< WModuleInputData< WDataSetScalar > > m_inputScalar;
	boost::shared_ptr< WModuleOutputData< WDataSetFibers > > m_output;
	boost::shared_ptr< WModuleOutputData< WDataSetScalar > > m_outputScalar;

	/* properties */
	WPropDouble m_xValue;
    WPropDouble m_yValue;
    WPropDouble m_zValue;

	WPropInt m_order;
	WPropInt m_reconstructions;
	WPropInt m_tesselationLevel;

    WPropTrigger m_aTrigger;
    WPropTrigger m_TriggerStartCalculate;
    WPropTrigger m_pushToCache;
    WPropTrigger m_clearCache;
    WPropColor m_color;

    /* strategies */
    WStrategyHelper< WObjectNDIP< WQBallAlgorithm > > m_strategy;
    WQBallCPU::SPtr m_qBallCPU;
    WQBallGPU::SPtr m_qBallGPU;

    /* drawing */
    osg::ref_ptr< osg::PositionAttitudeTransform > m_sphereTransformNode;

	boost::shared_ptr< WDataSetRawHARDI > datasetRaw;
	boost::shared_ptr< WDataSetScalar > datasetScalar;

	boost::shared_ptr< WCondition > m_propCondition;

	boost::shared_ptr< WGridRegular3D > m_grid;
	boost::shared_ptr< WGridRegular3D > m_gridScalar;
	std::vector< int > m_shOrders;

	std::vector< WUnitSphereCoordinates< double > > m_orientations;
	std::vector< WUnitSphereCoordinates< double > > m_reconstructionPoints;

	WMatrix< double > m_baseMatrix = WMatrix< double >(1);
	WMatrix< double > m_inverse = WMatrix< double >(1);
	WMatrix< double > m_pInverse = WMatrix< double >(1);
	WMatrix< double > m_reconstructionMatrix = WMatrix< double >(1);
	WMatrix< double > m_p = WMatrix< double >(1);
	WMatrix< double > m_odf = WMatrix< double >(1);
	WMatrix< double > m_coeffBase = WMatrix< double >(1);
	WMatrix< double > m_h = WMatrix< double >(1);

	WMatrixFixed< double, 3, 3 > m_transformation;

	WMatrixFixed< double, 3, 1 > m_pos;

	std::vector< std::vector< unsigned int > > m_neighbors;

	std::vector< WVector3d > sphereDistribution();
	void prepareMatrices();
	void calcLegendreMatrix();

	int fac(int a) {
		int b = 1;
		for(int i=1; i<=a; i++) {
			b = b * i;
		}
		return b;
	}

	float angle( WPosition p1, WPosition p2 ) {
		float frac = dot( p1, p2 ) / ( length( p1 ) * length( p2 ) );
		if( frac > 1.0 ) frac = 1.0;
		if( frac < -1.0 ) frac = -1.0;
		return acos( frac ) * 180 / M_PI;
	}
};

#endif
