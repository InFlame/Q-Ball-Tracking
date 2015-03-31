#ifndef WQBALLALGORITHM_H
#define WQBALLALGORITHM_H

#include <utility>

#include "core/dataHandler/WDataSetRawHARDI.h"
#include "core/dataHandler/WDataSetScalar.h"
#include "core/dataHandler/WDataSetFibers.h"
#include "core/dataHandler/WDataHandler.h"
#include "core/dataHandler/WFiberAccumulator.h"
#include "core/kernel/WKernel.h"
#include "core/common/math/WMath.h"
#include "core/common/math/WUnitSphereCoordinates.h"
#include "core/common/math/linearAlgebra/WMatrixFixed.h"


class WQBallAlgorithm
{
public:
	typedef WQBallAlgorithm super;

	virtual ~WQBallAlgorithm();

	virtual void prepareAlgorithm( boost::shared_ptr< WDataSetRawHARDI > datasetRaw, boost::shared_ptr< WDataSetScalar > datasetScalar, boost::filesystem::path path, boost::shared_ptr< WProgress > prog );
	virtual std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > operator()( boost::shared_ptr< WProgress > prog, WPosition startPosition );
	virtual void cleanUp();

	virtual int getSeedSize();
	
	bool isReady();

	void setOdf( WMatrix< double > odf );
	void setBase( WMatrix< double > base );
	void setCoeffBase( WMatrix< double > cBase );
	void setRegularization( WMatrix< double > h );
	void setNeighbors( std::vector< std::vector < unsigned int > > neighbors );
	void setOrientations( std::vector< WUnitSphereCoordinates< double > > reconstructionPoints );
	void setSHOrder( uint shOrder );

	void setClearCache( bool c );
	void clearScalarData();

protected:
	boost::shared_ptr< WDataSetRawHARDI > datasetRaw;
	boost::shared_ptr< WDataSetScalar > datasetScalar;

	boost::shared_ptr< WGridRegular3D > m_grid;
	boost::shared_ptr< WGridRegular3D > m_gridScalar;

	boost::filesystem::path m_path;

	bool m_ready;
	bool m_clearCache;

	boost::shared_ptr< WCondition > m_propCondition;

	WFiberAccumulator m_fiberAcc;

	boost::shared_ptr< WValueSet< double > > m_scalars;
	boost::shared_ptr< std::vector< double > > m_scalarData;

	// matrices
	WMatrix< double > m_odf = WMatrix< double >(1);
	WMatrix< double > m_baseMatrix = WMatrix< double >(1);
	WMatrix< double > m_coeffBase = WMatrix< double >(1);
	WMatrix< double > m_h = WMatrix< double >(1);
	std::vector< std::vector< unsigned int > > m_neighbors;
	std::vector< WPosition > m_reconstructionPoints;

	uint m_shOrder;

	std::vector< size_t > m_nonZeroGradients;

	WMatrixFixed< double, 3, 3 > m_transformation;

	virtual WValue< double > drawSample( WPosition );
	virtual boost::shared_ptr< WFiber > algorithm( WPosition );

};

#endif