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

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# General CMake Setup
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( QBallTractToolbox )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Include the OpenWalnut build system.
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# where to find our utils and cmake modules?
SET( OW_TOOLS_DIR ${PROJECT_SOURCE_DIR}/../tools )

# append search path for FindModules:
LIST( APPEND CMAKE_MODULE_PATH ${OW_TOOLS_DIR}/cmake )

# These scripts contains all the needed tools to setup the build:
# * Compiler Setup
# * Common OpenWalnut Options
# * Third-party Dependencies Setup
# * Unit Testing Setup if found
# * Doxygen Setup if found
INCLUDE( OpenWalnut )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Setup your additional dependencies and tools
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# IMPORTANT: NEVER add any commands bore the line INCLUDE( OpenWalnut ). This might cause problems!

FIND_PACKAGE( OpenMP )
MESSAGE( "OpenMP found: " ${OPENMP_FOUND} )
IF( OPENMP_FOUND )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
ENDIF()

FIND_PACKAGE( OpenCL )
MESSAGE( "OpenCL found: " ${OPENCL_FOUND} )
MESSAGE( STATUS "OpenCL Lib Used:" ${OPENCL_LIBRARY} )
IF( OPENCL_FOUND )
    INCLUDE_DIRECTORIES( AFTER ${OPENCL_INCLUDE_DIR} )
    #INCLUDE_DIRECTORIES( ${OPENCL_CPP_INCLUDE_DIR} )
    #TARGET_LINK_LIBRARIES( ${OPENCL_LIBRARY} )
ENDIF()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Setup all modules
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Here, you can add your own targets, call ADD_MODULE provided by OpenWalnut or a combination of it. When using ADD_MODULE or SETUP_MODULE, the
# OpenWalnut CMake scripts manages proper resource handling, test compilation and similar. You should use it whenever possible. More details on
# how to develop external modules, have a look at http://www.openwalnut.org. You can have a look at OpenWalnut.cmake if you want to modify
# the default behaviour.

# Please see the documentation of SETUP_MODULE as it shows how to add custom dependencies (third-party libs)

SETUP_MODULE( 
    ${PROJECT_NAME}   # use project name as module(-toolbox) name
    "."               # where to find the sources
    "${OPENCL_LIBRARY}"                # no additonal dependencies
    ""                # no sources to exclude
)

# Tell CMake that we want to use the WToolboxVersion header in our Modules. It gets created automatically if you keep this line.
SETUP_USE_VERSION_HEADER( ${PROJECT_NAME} )

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#
# Additional Setup
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Add the files under ../resources/MyGlobalResources as additional, global resources
# The first parameter is the directory name, the second parameter is the component name. This is used during installation to identify to which
# install component this belongs. The name is arbitrary but must be unique.
#
# NOTE: The first parameter of SETUP_MODULE is not allowed as component name here because SETUP_MODULE already registers this component. Simply
# append some string.
SETUP_GLOBAL_RESOURCES( "MyGlobalResources" ${PROJECT_NAME}_Global )
