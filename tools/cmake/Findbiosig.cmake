# This script searches biosig. See http://biosig.sourceforge.net
#
# The following variables will be filled:
#   * BIOSIG_FOUND - if header and lib was found
#   * BIOSIG_INCLUDE_DIR - the path of header if found
#   * BIOSIG_LIBRARY - the path to the library
#

FIND_PATH( BIOSIG_INCLUDE_DIR biosig.h /usr/include /usr/local/include )

# This hack is inspired by FindBoost.cmake. It ensures that only shared objects are found. Linking a SO with a static lib is not possible 
# in Linux. On other systems, this should be no problem.
SET( _ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
IF( CMAKE_HOST_SYSTEM MATCHES "Linux" )
    SET( CMAKE_FIND_LIBRARY_SUFFIXES .so )
ENDIF()
FIND_LIBRARY( BIOSIG_LIBRARY NAMES biosig PATH /usr/lib /usr/local/lib )
SET( CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )

SET( BIOSIG_FOUND FALSE )
IF( BIOSIG_INCLUDE_DIR AND BIOSIG_LIBRARY )
    SET( BIOSIG_FOUND TRUE )
ENDIF()

IF( BIOSIG_FOUND )
   IF( NOT biosig_FIND_QUIETLY )
       MESSAGE( STATUS "Found biosig: ${BIOSIG_LIBRARY} and include in ${BIOSIG_INCLUDE_DIR}" )
   ENDIF()
ELSE()
   IF( biosig_FIND_REQUIRED )
      MESSAGE( FATAL_ERROR "Could not find biosig." )
   ENDIF()
ENDIF()

