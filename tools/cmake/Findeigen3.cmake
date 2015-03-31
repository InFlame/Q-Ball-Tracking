# This script searches eigen3. See http://eigen.tuxfamily.org
#
# The following variables will be filled:
#   * EIGEN3_FOUND - if Eigen (header) was found
#   * EIGEN3_INCLUDE_DIR - the path of Eigne header if found
#
# You can set the environment variable "EIGEN3_INCLUDE_DIR" to help this script finding it if
# you placed it in some unusual location.

FIND_PATH( EIGEN3_INCLUDE_DIR Eigen/Core $ENV{EIGEN3_INCLUDE_DIR} 
                                         $ENV{HOME}/.local/include/eigen3
                                         /usr/include/eigen3
                                         /usr/local/include/eigen3 
                                         /opt/local/include/eigen3
                                         /sw/include/eigen3 )

SET( EIGEN3_FOUND FALSE )
IF( EIGEN3_INCLUDE_DIR )
    SET( EIGEN3_FOUND TRUE )
ENDIF()

IF( EIGEN3_FOUND )
   IF( NOT eigen3_FIND_QUIETLY )
       MESSAGE( STATUS "Found eigen3: ${EIGEN3_INCLUDE_DIR}" )
   ENDIF()
ELSE()
   IF( eigen3_FIND_REQUIRED )
      MESSAGE( FATAL_ERROR "Could not find eigen3. Check your distribution or download from http://eigen.tuxfamily.org" )
   ENDIF()
ENDIF()

