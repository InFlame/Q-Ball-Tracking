#---------------------------------------------------------------------------
#
# Project: OpenWalnut ( http://www.openwalnut.org )
#
# Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
# For more information see http:#www.openwalnut.org/copying
#
# This file is part of OpenWalnut.
#
# OpenWalnut is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenWalnut is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with OpenWalnut. If not, see <http:#www.gnu.org/licenses/>.
#
#---------------------------------------------------------------------------

INCLUDE( OpenWalnutUtils )

# name of the core lib. Used when linking modules or other GUI
SET( OW_LIB_OPENWALNUT "openwalnut" )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# This is executed on inclusion. It sets up everything needed. Beginning with compiler and OpenWalnut options and the third-party dependencies.
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# guard against in-source builds
IF( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
  MESSAGE( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. (you may need to remove CMakeCache.txt" )
ENDIF()

# the build types
IF( NOT CMAKE_BUILD_TYPE )
    SET( CMAKE_BUILD_TYPE Release
         CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo"
         FORCE
       )
ENDIF( NOT CMAKE_BUILD_TYPE )

# guard against typos in build-type strings
STRING( TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
IF( NOT cmake_build_type_tolower STREQUAL "debug" AND
    NOT cmake_build_type_tolower STREQUAL "release" AND
    NOT cmake_build_type_tolower STREQUAL "relwithdebinfo" AND
    NOT cmake_build_type_tolower STREQUAL "")
  MESSAGE( SEND_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo  and \"\" (case-insensitive).")
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# External Building Support (find OpenWalnut if needed)
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 1: some needed path setup for version headers. This is needed for searching OW.
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# where to put the header
SET( OW_VERSION_HEADER_DIRECTORY_RELATIVE "versionHeader" )

# this is the name of the INTERNAL OpenWalnut version header. It is needed for searching OW. Do not mix this up with the version header filename
# you use for the toolbox (external module building).
SET( OW_INTERNAL_VERSION_HEADER_FILENAME "core/WVersion.h" )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 2: start build system and either search OW or use it internally.
# ---------------------------------------------------------------------------------------------------------------------------------------------------

MESSAGE( STATUS "----------------------------------------------------------------------" )
MESSAGE( STATUS "Welcome! This is OpenWalnut Build System." )

# if this is included for external module building, find OpenWalnut.
IF( NOT ${OW_EXTERNAL_MODULE} )
    # to allow non-core code to access core and ext absolutely
    INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR} )
ELSE()
    MESSAGE( STATUS "Searching OpenWalnut ..." )   

    FIND_PATH( OPENWALNUT_INCLUDE_DIR core/kernel/WKernel.h ${OPENWALNUT_INCLUDEDIR} $ENV{OPENWALNUT_INCLUDEDIR} /usr/include/openwalnut /usr/local/include/openwalnut )
    FIND_LIBRARY( OPENWALNUT_LIBRARIES NAMES ${OW_LIB_OPENWALNUT} lib${OW_LIB_OPENWALNUT} HINTS 
                  ${OPENWALNUT_LIBDIR}
                  $ENV{OPENWALNUT_LIBDIR} 
                  /usr/lib
                  /usr/local/lib
                  /usr
                  /usr/local
                  /usr/local/openwalnut
                  /usr/local/OpenWalnut
                  $ENV{PROGRAMFILES}/OpenWalnut )

    # find the version header. There are two possibilities here: 
    # 1: we find it inside the include dir -- this is the case if we found an installed OW
    # 2: we find it inside the lib/../versionHeader dir -- this is the case if we found a local build of OW
    # NOTE: this relies on OPENWALNUT_INCLUDE_DIR which is filled above. If it is empty, no OW was found so it is not bad to not find the version
    # header.
    FIND_PATH( OPENWALNUT_VERSIONHEADER_DIR ${OW_INTERNAL_VERSION_HEADER_FILENAME} 
                # 1: we search in the hopefully found include dir:    
                ${OPENWALNUT_INCLUDE_DIR}
                # 2: we search in the build dir
                "$ENV{OPENWALNUT_LIBDIR}/../${OW_VERSION_HEADER_DIRECTORY_RELATIVE}" 
             )
    SET( OPENWALNUT_FOUND FALSE )

    # do not confuse the user with this
    MARK_AS_ADVANCED( FORCE OPENWALNUT_INCLUDE_DIR )
    MARK_AS_ADVANCED( FORCE OPENWALNUT_VERSIONHEADER_DIR )
    MARK_AS_ADVANCED( FORCE OPENWALNUT_LIBRARIES )
    MARK_AS_ADVANCED( FORCE OPENWALNUT_FOUND )

    # provide some output
    IF( OPENWALNUT_INCLUDE_DIR )
      MESSAGE( STATUS "Found OpenWalnut include files in ${OPENWALNUT_INCLUDE_DIR}." )
    ENDIF()
    IF( OPENWALNUT_VERSIONHEADER_DIR )
      MESSAGE( STATUS "Found OpenWalnut version header in ${OPENWALNUT_VERSIONHEADER_DIR}." )
    ENDIF()
    IF( OPENWALNUT_LIBRARIES )
      MESSAGE( STATUS "Found OpenWalnut libs in ${OPENWALNUT_LIBRARIES}." )
    ENDIF()

    # really found?
    IF( OPENWALNUT_INCLUDE_DIR AND OPENWALNUT_LIBRARIES AND OPENWALNUT_VERSIONHEADER_DIR )
        SET( OPENWALNUT_FOUND TRUE )
        MESSAGE( STATUS "Found OpenWalnut." )
    ELSE()
        MESSAGE( FATAL_ERROR "Could not find OpenWalnut. You can use the environment variables OPENWALNUT_INCLUDEDIR and  OPENWALNUT_LIBDIR to point to your OpenWalnut installation." )
    ENDIF()

    # include
    INCLUDE_DIRECTORIES( ${OPENWALNUT_INCLUDE_DIR} )
    INCLUDE_DIRECTORIES( ${OPENWALNUT_VERSIONHEADER_DIR} )
    SET( OW_LIB_OPENWALNUT ${OPENWALNUT_LIBRARIES} )
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 3: version header setup for the external module or OpenWalnut (depending on OW_EXTERNAL_MODULE)
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Generate needed headers
# NOTE: add a line ADD_DEPENDENCIES( XYZ OW_generate_version_header ) to your target XYZ if you need the header!

# Call the file differently depending on internal or external build
IF( NOT ${OW_EXTERNAL_MODULE} )
    # if we build OW, we want the version header to be placed in core later on. We use the internal version header filename directly
    SET( OW_VERSION_HEADER_FILENAME ${OW_INTERNAL_VERSION_HEADER_FILENAME} )
    SET( OW_VERSION_HEADER_PREFIX "W" )
ELSE()
    # if this is for external use, the module might want to use its own version header
    # if we build OW, we want the version header to be placed in core later on
    SET( OW_VERSION_HEADER_FILENAME "WToolboxVersion.h" )
    SET( OW_VERSION_HEADER_PREFIX "WTOOLBOX" )
ENDIF()

# the complete header filename:
SET( OW_VERSION_HEADER_DIRECTORY ${PROJECT_BINARY_DIR}/${OW_VERSION_HEADER_DIRECTORY_RELATIVE} )
SET( OW_VERSION_HEADER ${OW_VERSION_HEADER_DIRECTORY}/${OW_VERSION_HEADER_FILENAME} )

# to allow all those targets to find the header:
INCLUDE_DIRECTORIES( ${OW_VERSION_HEADER_DIRECTORY} )
# Setup the target
SETUP_VERSION_HEADER( ${OW_VERSION_HEADER} ${OW_VERSION_HEADER_PREFIX} )

# Set the OW version string. This can be used by others for setting target versions during compilation.
GET_VERSION_STRING( OW_VERSION OW_LIB_VERSION )
IF( NOT ${OW_EXTERNAL_MODULE} )
    MESSAGE( STATUS "OW Version: \"${OW_VERSION}\"; OW Lib Version: \"${OW_LIB_VERSION}\"." )
ELSE()
    MESSAGE( STATUS "OW Toolbox Version: \"${OW_VERSION}\"; OW Toolbox Lib Version: \"${OW_LIB_VERSION}\"." )
ENDIF()

# We need a SOVERSION too. This somehow describes the API compatibility. We use the major number here.
SPLIT_VERSION_STRING( ${OW_LIB_VERSION} OW_VERSION_MAJOR OW_VERSION_MINOR OW_VERSION_PATCH )
SET( OW_SOVERSION ${OW_VERSION_MAJOR} )

MESSAGE( STATUS "----------------------------------------------------------------------" )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# This is mandatory. Defines the target paths for building all elements in OpenWalnut
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# mimic layout of install dir for build:
# these dirs are the same for all parts of OW
SET( OW_RUNTIME_DIR_RELATIVE "bin" )
SET( OW_RUNTIME_DIR ${PROJECT_BINARY_DIR}/${OW_RUNTIME_DIR_RELATIVE} )
SET( OW_LIBRARY_DIR_RELATIVE "lib" )
SET( OW_LIBRARY_DIR ${PROJECT_BINARY_DIR}/${OW_LIBRARY_DIR_RELATIVE} )
SET( OW_ARCHIVE_DIR_RELATIVE "lib" )
SET( OW_ARCHIVE_DIR ${PROJECT_BINARY_DIR}/${OW_ARCHIVE_DIR_RELATIVE} )
SET( OW_MODULE_DIR_RELATIVE "lib/openwalnut" )
SET( OW_MODULE_DIR ${PROJECT_BINARY_DIR}/${OW_MODULE_DIR_RELATIVE} )
SET( OW_MAN_DIR_RELATIVE "share/man" )
SET( OW_MAN_DIR "${PROJECT_BINARY_DIR}/share/man" )

# One could also define this for every part of OW, but it does not make sense. It is "share" because it is shared among the different parts
SET( OW_SHARE_DIR_RELATIVE "share/openwalnut" )
SET( OW_SHARE_DIR ${PROJECT_BINARY_DIR}/${OW_SHARE_DIR_RELATIVE} )

# where to find the doxygen config
SET( OW_DOXYGEN_DIR ${PROJECT_SOURCE_DIR}/../doc/developer )

# set our paths for install targets
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OW_RUNTIME_DIR} )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OW_LIBRARY_DIR} )
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OW_ARCHIVE_DIR} )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Basic Build Setup Helpers
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# This method configures the needed compiler flags and system specific options. You probably do not need to call this explciitly. It is done in
# BUILD_SYSTEM_SETUP
FUNCTION( BUILD_SYSTEM_COMPILER )
    # Unfortunately libstdc++'s header files don't work with mingw in ansi mode (basically libstdc++'s fault)
    IF( CMAKE_HOST_SYSTEM MATCHES "Windows" )
        SET( CMAKE_CXX_FLAGS "-frtti -pedantic -Wall -Wno-long-long -Wextra " CACHE STRING "" FORCE )
    ELSE()
        SET( CMAKE_CXX_FLAGS "-frtti -pedantic -std=c++98 -Wall -Wno-long-long -Wextra " CACHE STRING "" FORCE )
    ENDIF()

    # Darwin's ld isn't GNU and doesn't like the following
    IF( NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" )
        ## The following allows us to prevent cyclic dependencies even on linux
        SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -Wl,--allow-shlib-undefined,--as-needed" CACHE STRING "" FORCE )
        SET( CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed" CACHE STRING "" FORCE )
    ENDIF()
    SET( CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "" FORCE )
    SET( CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -O0" CACHE STRING "" FORCE )
    SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DDEBUG -O2" CACHE STRING "" FORCE )

    # Allow injection of other flags
    # NOTE: do not set these variables somewhere in cmake. They are intended to be used when calling CMake from the command line.
    # Utilize this to append build flags from external systems (like dpkg-buildflags).
    SET( CMAKE_EXE_LINKER_FLAGS "${OW_LD_FLAGS_INJECT} ${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "" FORCE )
    SET( CMAKE_MODULE_LINKER_FLAGS "${OW_LD_FLAGS_INJECT} ${CMAKE_MODULE_LINKER_FLAGS}" CACHE STRING "" FORCE )
    SET( CMAKE_SHARED_LINKER_FLAGS "${OW_LD_FLAGS_INJECT} ${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "" FORCE )
    SET( CMAKE_CXX_FLAGS "${OW_CXX_FLAGS_INJECT} ${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE )
    SET( CMAKE_C_FLAGS "${OW_C_FLAGS_INJECT} ${CMAKE_C_FLAGS}" CACHE STRING "" FORCE )

    # This can be useful for debugging
    # MESSAGE( STATUS "CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}" )
    # MESSAGE( STATUS "CMAKE_MODULE_LINKER_FLAGS = ${CMAKE_MODULE_LINKER_FLAGS}" )
    # MESSAGE( STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}" )
    # MESSAGE( STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}" )
    # MESSAGE( STATUS "CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}" )

    # Supress compilation warnings from includes residing in system paths, see #230 for further details.
    SET( CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem" CACHE STRING "" FORCE )

ENDFUNCTION( BUILD_SYSTEM_COMPILER )

# GCC 4.7 requires us to explicitly link against libstdc++ and libm. CMake offers a variable for this called "CMAKE_STANDARD_LIBRARIES".
# Unfortunately, this variable is empty. We fill it here and hopefully this is fixed in the near future.
LIST( APPEND CMAKE_STANDARD_LIBRARIES "stdc++" "m" )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Compiler setup
# ---------------------------------------------------------------------------------------------------------------------------------------------------

BUILD_SYSTEM_COMPILER()
ADD_DEFINITIONS( ${OW_CPP_FLAGS_INJECT} )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# OpenWalnut specific options
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# OpenWalnut specific options

# sorry, linking not available properly on windows, Cygwin supports this but we do not want special rules for thousands of environments.
# ==> keep it clean
IF( NOT CMAKE_HOST_SYSTEM MATCHES "Windows" )
    OPTION( OW_LINK_SHADERS "If turned on, shaders will not be copied but will be linked. This is a nice option for developers." OFF )
ENDIF()

# Provide several options to control some aspects of resource copy.
OPTION( OW_PACKAGE_BUILD "Enable this to get fine-grained control over several resources and files getting installed. This is very handy for package building." OFF )
IF( OW_PACKAGE_BUILD )
    SET( OW_PACKAGE_PACKAGER "cpack" CACHE STRING "Package builder. Set this to enable packager-specific options during install." )

    OPTION( OW_PACKAGE_NOCOPY_LICENSE "Disable to copy our licensing information. Enabling this can be useful for package maintainer since several packaging systems have their own licence mechanism (i.e. Debian)." OFF )
    OPTION( OW_PACKAGE_NOCOPY_COREFONTS "Enable this if you have liberation fonts installed on your system. They will be linked. If disabled, our fonts are copied." OFF )
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# REQUIRED third party libs
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Boost, see http://www.boost.org/
#
# To see, which boost libs we currently use, you may run the following command # in the src directory on a linux box to make some investigations:
# grep -i include `find . -type f` | grep boost | awk '{print $2}' | sort | uniq

# Setup boost options
SET( Boost_USE_MULTITHREAD ON )
OPTION( BUILD_PYTHON_INTERPRETER OFF )

# find the boost packages
IF( BUILD_PYTHON_INTERPRETER )
    FIND_PACKAGE( Boost 1.46.0 REQUIRED program_options thread filesystem date_time system signals regex python )
ELSE()
    FIND_PACKAGE( Boost 1.46.0 REQUIRED program_options thread filesystem date_time system signals regex )
ENDIF() #BUILD_SCRIPTENGINE

# include the boost headers
INCLUDE_DIRECTORIES( SYSTEM ${Boost_INCLUDE_DIR} )

# avoid filesystem 2 stuff
ADD_DEFINITIONS( "-DBOOST_FILESYSTEM_VERSION=3" )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# OpenGL, at least 1.2
# See http://www.opengl.org

# Find OpenGL or OpenGL ES on Android
IF( ANDROID )
  # on Android, we rely on the fact the the GLES headers reside in the correct NDK search paths
  # -> so we do not add the include dir directly. We only set the variables needed to fake a found OpenGL:
  SET( OPENGL_FOUND ON )
  # link against GLES 2
  SET( OPENGL_LIBRARIES "GLESv2" )
  SET( OPENGL_gl_LIBRARY "GLESv2" )
ELSE()
  FIND_PACKAGE( OpenGL REQUIRED )
  # include the OpenGL header paths
  INCLUDE_DIRECTORIES( SYSTEM ${OPENGL_INCLUDE_DIR} )
ENDIF()

# -----------------------------------------------------------------------------------------------------------------------------------------------
# OpenSceneGraph, at least 2.8.0
# See http://www.openscenegraph.org

# find the needed packages
SET( MIN_OSG_VERSION 2.8.0 )
FIND_PACKAGE( OpenSceneGraph ${MIN_OSG_VERSION} REQUIRED osgWidget osgViewer osgText osgSim osgGA osgDB osgUtil )
IF( OPENSCENEGRAPH_FOUND )
    INCLUDE_DIRECTORIES( SYSTEM ${OPENSCENEGRAPH_INCLUDE_DIRS} )

    # is the OSG linked statically? If yes, we need to take care that all the osgdb plugins are linked too.
    STRING( REGEX MATCH "osgDB\\.a" OPENSCENEGRAPH_STATIC "${OPENSCENEGRAPH_LIBRARIES}" )
    IF( OPENSCENEGRAPH_STATIC )
      MESSAGE( STATUS "Found static OpenSceneGraph! Adding all osgDB plugins." )
      ADD_DEFINITIONS( -DOSG_LIBRARY_STATIC )

      # we need to find the path. Unfortunately, the OSG find scripts do not provide these info
      LIST( GET OPENSCENEGRAPH_LIBRARIES 0, firstElement )
      GET_FILENAME_COMPONENT( firstElementPath ${firstElement} PATH )

      # hopefully all static osgdb plugins reside in this directory
      FILE( GLOB plugInLibs "${firstElementPath}/*osgdb_*.a" )
      LIST( SORT plugInLibs )
      SET( OPENSCENEGRAPH_OSGDB_LIBS "" )
      FOREACH( plugIn ${plugInLibs} )
        LIST( APPEND OPENSCENEGRAPH_OSGDB_LIBS ${plugIn} )
      ENDFOREACH()

      # The osgDB plugins need to be linked BEFORE osg and osgDB
      LIST( INSERT OPENSCENEGRAPH_LIBRARIES 0 ${OPENSCENEGRAPH_OSGDB_LIBS} )
    ENDIF()
ENDIF()

# NOTE: sorry but this is needed since those vars spam the ccmake. The user should find the important options fast!

# Remove all of them
MARK_AS_ADVANCED( FORCE OSG_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSG_LIBRARY )
MARK_AS_ADVANCED( FORCE OSG_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGDB_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGDB_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGDB_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGGA_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGGA_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGGA_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGSIM_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGSIM_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGSIM_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGUTIL_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGUTIL_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGUTIL_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGTEXT_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGTEXT_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGTEXT_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGWIDGET_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGWIDGET_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGWIDGET_LIBRARY_DEBUG )
MARK_AS_ADVANCED( FORCE OSGVIEWER_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OSGVIEWER_LIBRARY )
MARK_AS_ADVANCED( FORCE OSGVIEWER_LIBRARY_DEBUG )

MARK_AS_ADVANCED( FORCE OPENTHREADS_INCLUDE_DIR )
MARK_AS_ADVANCED( FORCE OPENTHREADS_LIBRARY )
MARK_AS_ADVANCED( FORCE OPENTHREADS_LIBRARY_DEBUG )

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Eigen3, at least 3.0.0
# See http://eigen.tuxfamily.org

FIND_PACKAGE( eigen3 REQUIRED )
IF( EIGEN3_FOUND )
    INCLUDE_DIRECTORIES( SYSTEM ${EIGEN3_INCLUDE_DIR} )

    # NOTE: this is included in ext. But we need to set several definitions to make this work on 32 Bit machines due to alignment problems
    SET( EIGEN3_DEFINES -DEIGEN_DONT_VECTORIZE -DEIGEN_DONT_ALIGN -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT )
    ADD_DEFINITIONS( ${EIGEN3_DEFINES} )
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Unit Testing
#
#  - We use cxxtest. See http://cxxtest.tigris.org
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Try to find it
FIND_PACKAGE( CxxTest QUIET )

# If it is found, and OW_USE_TESTS is on, build the tests and activate CMake's test mechanisms
IF( CXXTEST_FOUND )
  # To enable testing
  OPTION( OW_USE_TESTS "This enables compilation of tests" ON )
  INCLUDE_DIRECTORIES( SYSTEM ${CXXTEST_INCLUDE_DIRS} )
  INCLUDE_DIRECTORIES( SYSTEM ${CXXTEST_INCLUDE_DIR} ) # NOTE: old FindCXXTest versions used this name
  IF( OW_USE_TESTS )
    SET( OW_COMPILE_TESTS ON ) #We need this variable because this is tested more often.
    # Package settings:
    SET( CXXTEST_USE_PYTHON 1 )
    # Activate CTest and "test" target
    ENABLE_TESTING()

    # we want a more verbose testing too.
    ADD_CUSTOM_TARGET( vtest
                       COMMAND $(MAKE) test ARGS="-V"
                       COMMENT "Runs the test in verboseness to see what actually went wrong"
                     )
  ELSE()
    SET( OW_COMPILE_TESTS OFF )
  ENDIF()
ELSE()
  SET( OW_COMPILE_TESTS OFF )
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Documentation
#  - Call doxygen here
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

IF( EXISTS ${OW_DOXYGEN_DIR}/doxygenConfig )
    #let doxygen do the work
    ADD_CUSTOM_TARGET( doc
                       COMMAND doxygen ${OW_DOXYGEN_DIR}/doxygenConfig
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..
                       COMMENT "Build doxygen documentation"
                       VERBATIM
                     )
ENDIF()

# On Windows, we want the common doc (README, AUTHORS and COPYING) to be in the root install dir too:
IF( CMAKE_HOST_SYSTEM MATCHES "Windows" )
    SETUP_COMMON_DOC( "." "COMMON_DOC_ON_WINDOWS" )
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Style
#  - We use brainlint for this
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# SETUP_STYLECHECKER adds targets for each lib which then add theirself to this target
ADD_CUSTOM_TARGET( stylecheck )

