#ifndef WQBALLCPU_H
#define WQBALLCPU_H

#include "core/kernel/WKernel.h"
#include "core/common/WObjectNDIP.h"
#include "core/common/WProperties.h"
#include "core/dataHandler/WFiberAccumulator.h"
#include "core/dataHandler/WDataSetRawHARDI.h"
#include "core/dataHandler/WDataSetScalar.h"

#include "WQBallAlgorithm.h"

#include "boost/random.hpp"

class WQBallCPU: public WObjectNDIP< WQBallAlgorithm >
{

public:

	WQBallCPU();
	WQBallCPU( bool parallel );
	virtual ~WQBallCPU();

	virtual void prepareAlgorithm( boost::shared_ptr< WDataSetRawHARDI > datasetRaw, boost::shared_ptr< WDataSetScalar > datasetScalar, boost::filesystem::path path, boost::shared_ptr< WProgress > prog );
	virtual std::pair<boost::shared_ptr< WDataSetFibers >, boost::shared_ptr< WDataSetScalar > > operator()( boost::shared_ptr< WProgress > prog, WPosition startPpos );

	virtual int getSeedSize();
	
private:
	WPropBool m_enableSH;
	WPropBool m_mainDir;

	WPropInt m_cores;

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

    float fa;
    float curv;
    float stepSize;
    float percentRange;
    int degreeRange;
    int stepDistance;


    bool m_parallel;

    boost::random::taus88 rndGen;
    boost::random::uniform_int_distribution<> rndDist;

	void getPeaks( WValue< double > &s, std::vector< int > &p );
	void drawSample( WPosition &position, WValue< double > &retValue );

	virtual boost::shared_ptr< WFiber >algorithm( WPosition position );

	float angle( WPosition &p1, WPosition &p2 ) {
		float frac = dot( p1, p2 ) / ( length( p1 ) * length( p2 ) );
		if( frac > 1.0 ) frac = 1.0;
		if( frac < -1.0 ) frac = -1.0;
		return acos( frac ) * 180 / M_PI;
	}
};

#endif
