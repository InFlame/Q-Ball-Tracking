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

# This function converts a given filename to a proper target name. This is very useful if you want to define custom targets for
# files and need a unique name.
# _filename the filename to convert
# _target returns the proper target string
FUNCTION( FILE_TO_TARGETSTRING _filename _target )
    # strip the whole path up to src
    STRING( REGEX REPLACE "^.*/src" "src" fileExcaped "${_filename}" ) 

    # remove all those ugly chars
    STRING( REGEX REPLACE "[^A-Za-z0-9]" "X" fileExcaped "${fileExcaped}" )

    # done. Return value
    SET( ${_target} "${fileExcaped}" PARENT_SCOPE )
ENDFUNCTION( FILE_TO_TARGETSTRING )

# Gets the major, minor and patch number from a version string.
# _VersionString the string to split
# _Major the major version number - result
# _Minor the minor version number - result
# _Patch the patch number - result
FUNCTION( SPLIT_VERSION_STRING _VersionString _Major _Minor _Patch )
  STRING( STRIP _VersionString ${_VersionString} )
  STRING( REGEX MATCH "^[0-9]+" _MajorProxy "${_VersionString}" )
  STRING( REGEX MATCH "[^0-9][0-9]+[^0-9]" _MinorProxy "${_VersionString}" )
  STRING( REGEX MATCH "[0-9]+" _MinorProxy "${_MinorProxy}" )
  STRING( REGEX MATCH "[0-9]+$" _PatchProxy "${_VersionString}" )
  SET( ${_Major} "${_MajorProxy}" PARENT_SCOPE )
  SET( ${_Minor} "${_MinorProxy}" PARENT_SCOPE )
  SET( ${_Patch} "${_PatchProxy}" PARENT_SCOPE )
ENDFUNCTION( SPLIT_VERSION_STRING )

# Recursively searches compile files (headers, sources).
# _DirString: where to search
# _CPPFiles contains the cpp files afterwards
# _HFiles contains the h files afterwards, without tests
# _TestFiles contains only the test headers
FUNCTION( COLLECT_COMPILE_FILES _DirString _CPPFiles _HFiles _TestFiles )
    # recursively get all files
    FILE( GLOB_RECURSE CPP_FILES ${_DirString}/*.cpp )
    FILE( GLOB_RECURSE H_FILES   ${_DirString}/*.h )
    FILE( GLOB_RECURSE TEST_FILES   ${_DirString}/*_test.h )

    # exclude some special dirs
    FOREACH( file ${H_FILES} )
        # the test directories should be excluded from normal compilation completely
        STRING( REGEX MATCH "^.*\\/test\\/.*" IsTest "${file}" )
        # ext sources should be build seperatly 
        STRING( REGEX MATCH "^.*\\/ext\\/.*" IsExternal "${file}" )
        IF( IsTest )
            LIST( REMOVE_ITEM H_FILES ${file} )
        ENDIF()
        IF( IsExternal )
            LIST( REMOVE_ITEM H_FILES ${file} )
        ENDIF()
    ENDFOREACH( file )
    FOREACH( file ${CPP_FILES} )
        # the test directories should be excluded from normal compilation completely
        STRING( REGEX MATCH "^.*\\/test\\/.*" IsTest "${file}" )
        # ext sources should be build seperatly 
        STRING( REGEX MATCH "^.*\\/ext\\/.*" IsExternal "${file}" )
        IF( IsTest )
            LIST( REMOVE_ITEM CPP_FILES ${file} )
        ENDIF()
        IF( IsExternal )
            LIST( REMOVE_ITEM CPP_FILES ${file} )
        ENDIF()
    ENDFOREACH( file )

    SET( ${_CPPFiles} "${CPP_FILES}" PARENT_SCOPE )
    SET( ${_HFiles} "${H_FILES}" PARENT_SCOPE )
    SET( ${_TestFiles} "${TEST_FILES}" PARENT_SCOPE )
ENDFUNCTION( COLLECT_COMPILE_FILES )

# Recursively searches shader files with extension glsl.
# _DirString where to search
# _GLSLFiles the list of files found
FUNCTION( COLLECT_SHADER_FILES _DirString _GLSLFiles )
    # recursively get all files
    FILE( GLOB_RECURSE GLSL_FILES ${_DirString}/*.glsl )
    SET( ${_GLSLFiles} "${GLSL_FILES}" PARENT_SCOPE )
ENDFUNCTION( COLLECT_SHADER_FILES )

# Add tests and copy their fixtures by specifiying the test headers. This is completely automatic and is done recusrively (fixtures). The test
# filenames need to be absolute. The fixtures are searched in CMAKE_CURRENT_SOURCE_DIR.
# _TEST_FILES the list of test header files.
# _TEST_TARGET for which target are the tests? This is added as link library for each test too.
# Third, unnamed parameter: additional dependencies as list
FUNCTION( SETUP_TESTS _TEST_FILES _TEST_TARGET )
    # Only do something if needed
    IF( OW_COMPILE_TESTS )
        # Abort if no tests are present
        LIST( LENGTH _TEST_FILES TestFileListLength )
        IF( ${TestFileListLength} STREQUAL "0" )
            # MESSAGE( STATUS "No tests for target ${_TEST_TARGET}. You should always consider unit testing!" )
            RETURN()
        ENDIF()

        # the optional parameter is an additional dependencies list
        SET( _DEPENDENCIES ${ARGV2} )

        # target directory for fixtures
        SET( FixtureTargetDirectory "${CMAKE_BINARY_DIR}/fixtures/${_TEST_TARGET}/" )

        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Setup CXX test
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # Also create a list of paths where the tests where found
        SET( FixturePaths )

        # Add each found test and create a target for it
        FOREACH( testfile ${_TEST_FILES} )
            # strip path and extension
            STRING( REGEX REPLACE "^.*/" "" StrippedTestFile "${testfile}" )
            STRING( REGEX REPLACE "\\..*$" "" PlainTestFileName "${StrippedTestFile}" )
            STRING( REGEX REPLACE "_test$" "" TestFileClass "${PlainTestFileName}" )

            # also extract test path
            STRING( REPLACE "${StrippedTestFile}" "fixtures" TestFileFixturePath "${testfile}" )
            IF( EXISTS "${TestFileFixturePath}" AND IS_DIRECTORY "${TestFileFixturePath}" )
                LIST( APPEND FixturePaths ${TestFileFixturePath} )
            ENDIF()

            # create some name for the test
            SET( UnitTestName "unittest_${TestFileClass}" )

            # create the test-target
            CXXTEST_ADD_TEST( ${UnitTestName} "${UnitTestName}.cc" ${testfile} )
            TARGET_LINK_LIBRARIES( ${UnitTestName} ${_TEST_TARGET} ${CMAKE_STANDARD_LIBRARIES} ${_DEPENDENCIES} )

            # unfortunately, the tests search their fixtures relative to their working directory. So we add a preprocessor define containing the
            # path to the fixtures. This is quite ugly but I do not know how to ensure that the working directory of tests can be modified.
            SET_SOURCE_FILES_PROPERTIES( "${UnitTestName}.cc" PROPERTIES COMPILE_DEFINITIONS "W_FIXTURE_PATH=std::string( \"${FixtureTargetDirectory}\" )" )
        ENDFOREACH( testfile )

        # -------------------------------------------------------------------------------------------------------------------------------------------
        # Search fixtures
        # -------------------------------------------------------------------------------------------------------------------------------------------

        # REMOVE_DUPLICATES throws an error if list is empty. So we check this here
        LIST( LENGTH FixturePaths ListLength )
        IF( NOT ${ListLength} STREQUAL "0" )
            # the list may contain duplicates
            LIST( REMOVE_DUPLICATES FixturePaths )

            # ---------------------------------------------------------------------------------------------------------------------------------------
            # Create copy target for each fixture directory
            # ---------------------------------------------------------------------------------------------------------------------------------------

            # for each fixture, copy to build dir
            FOREACH( FixtureDir ${FixturePaths} )
                # we need a unique name for each fixture dir as target
                FILE_TO_TARGETSTRING( ${FixtureDir} FixtureDirEscaped )

                # finally, create the copy target
                ADD_CUSTOM_TARGET( ${_TEST_TARGET}_CopyFixtures_${FixtureDirEscaped}
                    COMMAND ${CMAKE_COMMAND} -E copy_directory "${FixtureDir}" "${FixtureTargetDirectory}"
                    COMMENT "Copy fixtures of ${_TEST_TARGET} from ${FixtureDir} to ${FixtureTargetDirectory}."
                )
                ADD_DEPENDENCIES( ${_TEST_TARGET} ${_TEST_TARGET}_CopyFixtures_${FixtureDirEscaped} )
            ENDFOREACH( FixtureDir )
        ENDIF()
    ENDIF( OW_COMPILE_TESTS )
ENDFUNCTION( SETUP_TESTS )

# This function sets up the build system to ensure that the specified list of shaders is available after build in the target directory. It
# additionally setups the install targets. Since build and install structure are the same, specify only relative targets here which are used for
# both.
# _Shaders list of shaders
# _TargetDir the directory where to put the shaders. Relative to ${PROJECT_BINARY_DIR} and install dir. You should avoid ".." stuff. This can
# break the install targets
# _Component the name of the install component
FUNCTION( SETUP_SHADERS _Shaders _TargetDir _Component )
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E make_directory ${_TargetDir} )

    # should we copy or link them?
    SET( ShaderOperation "copy_if_different" )
    IF ( OW_LINK_SHADERS ) # link
         SET( ShaderOperation "create_symlink" )
    ENDIF( OW_LINK_SHADERS )

    # now do the operation for each shader into build dir
    FOREACH( fname ${_Shaders} )
        # We need the plain filename (create_symlink needs it)
        STRING( REGEX REPLACE "^.*/" "" StrippedFileName "${fname}" )

        # construct final filename
        SET( targetFile "${PROJECT_BINARY_DIR}/${_TargetDir}/${StrippedFileName}" )

        # if the file already exists as non-symlink and we use the "create_symlink" command, we first need to remove the files
        IF( NOT IS_SYMLINK ${targetFile} AND OW_LINK_SHADERS )
            # before creating the symlink, remove the old files or cmake will not create the symlinks (as intended)
            EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E remove -f ${targetFile} )
        ELSEIF( IS_SYMLINK ${targetFile} AND NOT OW_LINK_SHADERS )
            # also handle the inverse case. The files exist as symlinks but copies should be used. Remove symlinks!
            EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E remove -f ${targetFile} )
        ENDIF()

        # let cmake do it
        EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E ${ShaderOperation} ${fname} ${targetFile} )
    ENDFOREACH( fname )

    # now add install targets for each shader. All paths are relative to the current source dir.
    FOREACH( fname ${_Shaders} )
        INSTALL( FILES ${fname} DESTINATION ${_TargetDir} 
                                COMPONENT ${_Component}
               )
    ENDFOREACH( fname )
ENDFUNCTION( SETUP_SHADERS )

# Sets up the stylecheck mechanism. Use this to add your codes to the stylecheck mechanism.
# _TargetName the name of the target which gets added here
# _CheckFiles the list of files to check
# _Excludes a list of exclusion rules. These are regular expressions.
FUNCTION( SETUP_STYLECHECKER _TargetName _CheckFiles _Excludes )

    # to exlude some files, check each file against each exlusion rule
    FOREACH( filename ${_CheckFiles} )
        FOREACH( excludeRule ${_Excludes} )
            STRING( REGEX MATCH "${excludeRule}" IsExcluded "${filename}" )
            IF( IsExcluded )
                LIST( REMOVE_ITEM _CheckFiles ${filename} )
            ENDIF( IsExcluded )
        ENDFOREACH( excludeRule )
    ENDFOREACH( filename )

    # the stylechecker allows coloring the output. Enable if color make is active
    IF( CMAKE_COLOR_MAKEFILE )
        SET( STYLECHECK_OPTIONS "--color" )
    ELSE()
        SET( STYLECHECK_OPTIONS "" )
    ENDIF()

    # Further system specific options
    IF( CMAKE_HOST_WIN32 )
        SET( XARGS_OPTIONS "-n 128" )
    ELSEIF( CMAKE_HOST_UNIX )
        SET( XARGS_OPTIONS "-n 64" )
    ELSE()
        SET( XARGS_OPTIONS "" )
    ENDIF()

    # Export our filtered file list to a file in build dir
    SET( BrainLinterListFile "${PROJECT_BINARY_DIR}/brainlint/brainlintlist_${_TargetName}" )
    LIST( REMOVE_ITEM _CheckFiles "" )
    STRING( REPLACE ";" "\n" _CheckFiles "${_CheckFiles}" )
    FILE( WRITE ${BrainLinterListFile} "${_CheckFiles}\n" )

    # add a new target for this lib
    ADD_CUSTOM_TARGET( stylecheck_${_TargetName}
                       COMMAND  cat ${BrainLinterListFile} | xargs ${XARGS_OPTIONS} ${PROJECT_SOURCE_DIR}/../tools/style/brainlint/brainlint.py ${STYLECHECK_OPTIONS} 2>&1 | grep -iv 'Total errors found: 0$$' | cat
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       COMMENT "Check if ${_TargetName} complies to CodingStandard"
    )

    # make the stylecheck taret depend upon this
   ADD_DEPENDENCIES( stylecheck "stylecheck_${_TargetName}" )
ENDFUNCTION( SETUP_STYLECHECKER )

# This function handles local and global resources needed for program execution. This is a generic function which allows any resource in a source directory
# to be copied  to a target directory in build and install directory. Use this for module resources or the OpenWalnut resources.
#
# For convenience, this function first checks in _source_path, and if nothing was found there, in ${CMAKE_CURRENT_SOURCE_DIR}/${_source_path}]
# _source_path path to a directory whose CONTENTS get copied/installed.
# _destination_path where to put the resources. This MUST be relative to the install/build dir (${PROJECT_BINARY_DIR})
# _component the component to which this is related. This must be unique for every call.
# _installcomponent to which installation component does this resource belong? (i.e. MODULES ...)
FUNCTION( SETUP_RESOURCES_GENERIC _source_path _destination_path _component _installcomponent )
    # as all the resources with the correct directory structure reside in ../resources, this target is very easy to handle
    SET( ResourcesPath "${_source_path}" )

    # if the given dir does not exists here, it might be relative? lets try to check this against the current source directory
    IF( NOT IS_DIRECTORY "${ResourcesPath}" )
      SET( ResourcesPath ${CMAKE_CURRENT_SOURCE_DIR}/${ResourcesPath} )
    ENDIF()

    # remove trailing slashes if any
    STRING( REGEX REPLACE "/$" "" ResourcesPath "${ResourcesPath}" )
    # this ensures we definitely have one slash at the end
    SET( ResourcesPath "${ResourcesPath}/" )

    IF( IS_DIRECTORY "${ResourcesPath}" )
        MESSAGE( STATUS "Found resources in ${ResourcesPath}" )

        ADD_CUSTOM_TARGET( ResourceConfiguration_${_component}
            ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${ResourcesPath}" "${PROJECT_BINARY_DIR}/${_destination_path}/"
            COMMENT "Copying resources for ${_component} to build directory"
        )

        # Also specify install target
        INSTALL( DIRECTORY ${ResourcesPath}
                 DESTINATION "${_destination_path}"
                 COMPONENT ${_installcomponent}
                 PATTERN "bin/*"            # binaries need to be executable
                     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                 GROUP_READ GROUP_EXECUTE
                                 WORLD_READ WORLD_EXECUTE
                 )
    ENDIF()
ENDFUNCTION( SETUP_RESOURCES_GENERIC )

# This function handles global resources needed for program execution. Place your resources in "${CMAKE_CURRENT_SOURCE_DIR}/../resources/". They
# get copied to the build directory and a proper install target is provided too. 
# _component the component to which the resources belong, must be unique
# _resource the exact resource under resources-directory
FUNCTION( SETUP_GLOBAL_RESOURCES _resource _component )
    # just forward the call with the proper parameter
    SETUP_RESOURCES_GENERIC( "${PROJECT_SOURCE_DIR}/../resources/${_resource}/"  # source dir
                             "."                            # target dir - global resources always put into the build/install dir directly
                             ${_component}                  # component
                             ${_component}                  # install component is same as component as it is unique
    )

    # allow platform dependent setup
    STRING( TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM_NAME_LOWER )
    # just forward the call with the proper parameter
    SETUP_RESOURCES_GENERIC( "${PROJECT_SOURCE_DIR}/../resources/platformDependent/${_resource}/${PLATFORM_NAME_LOWER}"  # source dir
                             "."                            # target dir - global resources always put into the build/install dir directly
                             ${_component}_${PLATFORM_NAME_LOWER}  # component
                             ${_component}                  # install component is same as component as it is unique
    )

ENDFUNCTION( SETUP_GLOBAL_RESOURCES )

# This function copies the typical source docs (README, AUTHORS, CONTRIBUTORS and Licence files to the specified directory.
# _component the component to which the resources belong
# _targetDirRelative the relative target dir.
FUNCTION( SETUP_COMMON_DOC _target _component )
    SETUP_ADDITIONAL_FILES( ${_target} 
                            ${_component}
                            "${PROJECT_SOURCE_DIR}/../README"
                            "${PROJECT_SOURCE_DIR}/../AUTHORS"
                            "${PROJECT_SOURCE_DIR}/../CONTRIBUTORS"
                          )

    # If the user did not disable license-copying, do it
    # NOTE: use this "double-negative" to use the fact that undefined variables yield FALSE.
    IF( NOT OW_PACKAGE_NOCOPY_LICENSE )
        SETUP_ADDITIONAL_FILES( ${_target} 
                                ${_component}
                                "${PROJECT_SOURCE_DIR}/../COPYING"
                                "${PROJECT_SOURCE_DIR}/../COPYING.LESSER"
                              )
    ENDIF()
ENDFUNCTION( SETUP_COMMON_DOC )

# This function configures the soecified file in the given resource. This is especially useful for scripts where to replace certain
# CMake variables (like pkg-config files).
# _resource the name of the resource in the resources directory
# _file the file in the the resource to configure
# _component the install component to which the file belongs
FUNCTION( SETUP_CONFIGURED_FILE _resource _file _component )
    SET( ResourcesPath "${PROJECT_SOURCE_DIR}/../resources/${_resource}/" )
    CONFIGURE_FILE( "${ResourcesPath}/${_file}" "${PROJECT_BINARY_DIR}/${_file}" @ONLY )
    # Install the file
    GET_FILENAME_COMPONENT( filepath ${_file} PATH )
    INSTALL( FILES "${PROJECT_BINARY_DIR}/${_file}" DESTINATION "${filepath}"
                                                      COMPONENT ${_component}
           )
ENDFUNCTION( SETUP_CONFIGURED_FILE )

# This function eases the process of copying and installing additional files which not reside in the resource path.
# It creates a target (ALL is depending on it) AND the INSTALL operation.
# _destination where to put them. This MUST be relative to the build dir and install dir.
# _component the name of the component for these files
# _OTHERS you can add an arbitrary list of additional arguments which represent the files to copy.
FUNCTION( SETUP_ADDITIONAL_FILES _destination _component )
    FOREACH( _file ${ARGN} )
        # only do if it exists
        IF( EXISTS ${OW_VERSION_FILENAME} )
            # create useful target name
            FILE_TO_TARGETSTRING( ${_file} fileTarget )

            # add a copy target
            ADD_CUSTOM_TARGET( CopyAdditionalFile_${fileTarget}_${_component}
                ALL
                COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/${_destination}/"
                COMMAND ${CMAKE_COMMAND} -E copy "${_file}" "${PROJECT_BINARY_DIR}/${_destination}/"
                COMMENT "Copying file ${_file}"
            )

            # add a INSTALL operation for this file
            INSTALL( FILES ${_file} DESTINATION ${_destination}
                                    COMPONENT ${_component}
                   )
        ENDIF()
    ENDFOREACH() 
ENDFUNCTION( SETUP_ADDITIONAL_FILES )

# This function copies a given directory or its contents to the specified destination. Since cmake is quite strange in handling directories
# somehow, we needed to trick here. 
# _destination where to put the directory/its contents. Realtive to build dir and install dir.
# _directory the directory to copy
# _component the name of the component for these files
# _contents if TRUE, the contents of _directory are copied into _destination. If FALSE, _destination as-is is copied to _destination/.. (sorry
#           for this weird stuff. Complain at cmake mailing list ;-))
FUNCTION( SETUP_ADDITIONAL_DIRECTORY _destination _directory _component _contents )
    # create a nice target name
    FILE_TO_TARGETSTRING( ${_directory} directoryTarget )

    # add a copy target
    # this copies the CONTENTS of the specified directory into the specified destination dir.
    # NOTE: cmake -E says, that copying a directory with the copy command is pssible. But on my system it isn't.
    ADD_CUSTOM_TARGET( CopyAdditionalDirectory_${directoryTarget}_${_component}
        ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/${_destination}/"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_directory}" "${PROJECT_BINARY_DIR}/${_destination}"
        COMMENT "Copying directory ${_directory}"
    )

    # we need to distinquish here whether the user wants to copy the contents of the specified directory or the whole directory.
    # NOTE: unfortunately, the semantics of cmake -E and INSTALL are different. We need to fix this with this hack.
    IF( _contents )
        # OK, the user wants us to copy the contents of the specified _directory INTO the dpecified destination
        SET(  InstallDestination "${_destination}" )
    ELSE()
        # see "cmake -E " for help. The copy_directory copies its contents and copy copies the directory as is.
        SET(  InstallDestination "${_destination}/../" )
    ENDIF()

    # add a INSTALL operation for this file
    INSTALL( DIRECTORY ${_directory}
             DESTINATION ${InstallDestination}
             COMPONENT ${_component}
           )
ENDFUNCTION( SETUP_ADDITIONAL_DIRECTORY )

# Function to setup and library install target. It contains all the permission and namelink magic needed.
# _libName the library to install (needs to be a targed added with ADD_LIBRARY).
# _targetRelative the relative target dir. You should use OW_LIBRARY_DIR_RELATIVE in most cases
# _component the name of the component to which the lib belongs. If you use some strange things here, consider updating the package configuration
#            too as it uses these components
FUNCTION( SETUP_LIB_INSTALL _libName _targetRelative _component )
    # NOTE: we need two separate install targets here since the namelink to the lib (libopenwalnut.so -> linopenwalnut.so.1.2.3) is only needed
    # in the DEV release. Have a look at NAMELINK_SKIP and NAMELINK_ONLY
    INSTALL( TARGETS ${_libName}
                ARCHIVE # NOTE: this is needed on windows (*.dll.a)
                    DESTINATION ${_targetRelative} 
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
                                GROUP_READ GROUP_EXECUTE  
                                WORLD_READ WORLD_EXECUTE
                RUNTIME # NOTE: this is needed on windows (*.dll)
                    DESTINATION ${_targetRelative} 
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
                                GROUP_READ GROUP_EXECUTE  
                                WORLD_READ WORLD_EXECUTE
                LIBRARY # NOTE: this is needed for all the others
                    DESTINATION ${_targetRelative}
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                GROUP_READ GROUP_EXECUTE
                                WORLD_READ WORLD_EXECUTE
                    NAMELINK_SKIP
             COMPONENT ${_component}
    )
ENDFUNCTION( SETUP_LIB_INSTALL )

# Function which handles typical "developer" install targets. It creates the symlink (namelink) for the lib and copies the headers to some
# include directory.
# _libName the devel target for this lib (needs to be a target name created with ADD_LIBRARY)
# _targetRelative the relative path to the lib
# _headers a list of headers. (Lists are ;-separated). This function strips CMAKE_CURRENT_SOURCE_DIR from each path!
# _headerTargetRelative relative target dir for the includes
# _component the name of the component.
FUNCTION( SETUP_DEV_INSTALL _libName _targetRelative _headers _headerTargetRelative _component )
    INSTALL( TARGETS ${_libName}
                ARCHIVE # NOTE: this is needed on windows
                    DESTINATION ${_targetRelative} 
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                GROUP_READ GROUP_EXECUTE
                                WORLD_READ WORLD_EXECUTE
                RUNTIME # NOTE: this is needed on windows (*.dll)
                    DESTINATION ${_targetRelative} 
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
                                GROUP_READ GROUP_EXECUTE  
                                WORLD_READ WORLD_EXECUTE
                LIBRARY # NOTE: this is needed for all the others
                    DESTINATION ${_targetRelative}
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                GROUP_READ GROUP_EXECUTE
                                WORLD_READ WORLD_EXECUTE
                    NAMELINK_ONLY
             COMPONENT ${_component}
    )

    # we want to copy the headers to. Unfortunately, cmake's install command does not preserve the directory structure.
    FOREACH( _header ${${_headers}} )
        STRING( REGEX MATCH "(.*)[/\\]" directory ${_header} )
        STRING( REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" directoryRelative ${directory} )
        INSTALL( FILES ${_header} 
                    DESTINATION ${_headerTargetRelative}/${directoryRelative}
                    COMPONENT ${_component}
               )
    ENDFOREACH()
ENDFUNCTION( SETUP_DEV_INSTALL )

# This function tries to find a proper version string. It therefore uses the file src/../VERSION and mercurial. If the file exists and is not
# empty, the contents of it get combined with the mercurial results if mercurial is installed. If not, only the file content will be used. If
# both methods fail, a default string is used.
# _version the returned version string
# _api_version returns only the API-version loaded from the version file. This is useful to set CMake version info for release compilation
# _default a default string you specify if all version check methods fail
FUNCTION( GET_VERSION_STRING _version _api_version )
    # Undef the OW_VERSION variable
    UNSET( OW_VERSION_HG )
    UNSET( OW_VERSION_FILE )

    # Grab version file.
    SET( OW_VERSION_FILENAME ${PROJECT_SOURCE_DIR}/../VERSION )
    IF( EXISTS ${OW_VERSION_FILENAME} )
        # Read the version file
        FILE( READ ${OW_VERSION_FILENAME} OW_VERSION_FILE_CONTENT )
        # The first regex will mathc 
        STRING( REGEX REPLACE ".*[^#]VERSION=([0-9]+\\.[0-9]+\\.[0-9]+[_~a-z,A-Z,0-9]*(\\+hgX?[0-9]*)?).*" "\\1"  OW_VERSION_FILE  ${OW_VERSION_FILE_CONTENT} ) 
        STRING( COMPARE EQUAL ${OW_VERSION_FILE} ${OW_VERSION_FILE_CONTENT}  OW_VERSION_FILE_INVALID )
        IF( OW_VERSION_FILE_INVALID )
            UNSET( OW_VERSION_FILE )
        ENDIF()

        # this is ugly because, if the version file is empty, the OW_VERSION_FILE content is "". Unfortunately, this is not UNDEFINED as it would be
        # by SET( VAR "" ) ... so set it manually
        IF( OW_VERSION_FILE STREQUAL "" )
            UNSET( OW_VERSION_FILE )
        ENDIF()
    ENDIF()
    # if the version file could not be parsed, print error
    IF( NOT OW_VERSION_FILE )
        MESSAGE( FATAL_ERROR "Could not parse \"${PROJECT_SOURCE_DIR}/../VERSION\"." )
    ENDIF()

    # Use hg to query version information.
    # -> the nice thing is: if hg is not available, no compilation errors anymore
    # NOTE: it is run insde the project source directory
    EXECUTE_PROCESS( COMMAND hg parents --template "{rev}" OUTPUT_VARIABLE OW_VERSION_HG RESULT_VARIABLE hgParentsRetVar 
                     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    )
    IF( NOT ${hgParentsRetVar} STREQUAL 0 )
        UNSET( OW_VERSION_HG )
        # be more nice if we do not find mercruial version. The simply strip the +hg tag.
        STRING( REGEX REPLACE "\\+hgX" "" OW_VERSION ${OW_VERSION_FILE} )
    ELSE()
        # if we have the mercurial info -> complement the version string
        STRING( REGEX REPLACE "hgX" "hg${OW_VERSION_HG}" OW_VERSION ${OW_VERSION_FILE} )
    ENDIF()
  
    SET( ${_version} ${OW_VERSION} PARENT_SCOPE )

    # we need to separate the API version too. This basically is the same as the release version, but without the HG statement
    STRING( REGEX REPLACE "([0-9]+\\.[0-9]+\\.[0-9]).*" "\\1"  OW_API_VERSION ${OW_VERSION} ) 
    SET( ${_api_version} ${OW_API_VERSION} PARENT_SCOPE )

ENDFUNCTION( GET_VERSION_STRING )

# This functions adds a custom target for generating the specified version header. This is very useful if you want to include build-time version
# information in your code. If your executable or lib is created in the same CMakeLists as this function is called, just add the
# _OW_VERSION_Header filename you specified here to your compile files. If you call this and add target XYZ in a subdirectory, you NEED to add
# the lines: 
#  ADD_DEPENDENCIES( XYZ OW_generate_version_header )
#  SET_SOURCE_FILES_PROPERTIES( ${OW_VERSION_HEADER} PROPERTIES GENERATED ON )
# This is needed since CMake can only use the ADD_CUSTOM_COMMAND output as dependency if the custom command was called inside the SAME directory
# as the target is. If not, an additional target needs to be defined and CMake needs information about generated files.
#
# _OW_VERSION_HEADER the filename where to store the header. Should be absolute.
# _PREFIX the string used as prefix in the header. Useful if you have multiple version headers.
FUNCTION( SETUP_VERSION_HEADER _OW_VERSION_HEADER _PREFIX )
    # This ensures that an nonexisting .hg/dirstate file won't cause a compile error (do not know how to make target)
    SET( HG_DEP "" )
    IF( EXISTS ${PROJECT_SOURCE_DIR}/../.hg/dirstate )
        SET( HG_DEP ${PROJECT_SOURCE_DIR}/../.hg/dirstate )
    ENDIF()

    # The file WVersion.* needs the version definition.
    ADD_CUSTOM_COMMAND( OUTPUT ${_OW_VERSION_HEADER}
                        DEPENDS ${PROJECT_SOURCE_DIR}/../VERSION ${HG_DEP}
                        COMMAND ${CMAKE_COMMAND} -D PROJECT_SOURCE_DIR:STRING=${PROJECT_SOURCE_DIR} -D HEADER_FILENAME:STRING=${_OW_VERSION_HEADER} -D PREFIX:STRING=${_PREFIX} -P OpenWalnutVersion.cmake
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../tools/cmake/
                        COMMENT "Creating Version Header ${_OW_VERSION_HEADER}."
    )

    # this is needed if the file is also a dependency of an target XYZ in a CMakeLists file in some subdirectory. Then, you should add 
    # ADD_DEPENDENCIES( XYZ OW_generate_version_header )
    ADD_CUSTOM_TARGET( OW_generate_version_header DEPENDS ${_OW_VERSION_HEADER} )
ENDFUNCTION( SETUP_VERSION_HEADER )

# Comvenience function which does what SETUP_VERSION_HEADER requests from you if you need the version header in some CMakeLists different from
# the one which called SETUP_VERSION_HEADER. It uses ${OW_VERSION_HEADER} automatically. So it needs to be set.
#
# _target: the target name which needs the header
FUNCTION( SETUP_USE_VERSION_HEADER _target )
    SET_SOURCE_FILES_PROPERTIES( ${OW_VERSION_HEADER} PROPERTIES GENERATED ON )
    ADD_DEPENDENCIES( ${_target} OW_generate_version_header )
ENDFUNCTION( SETUP_USE_VERSION_HEADER )

# Automatically add a module. Do not use this function if your module nees additional dependencies or similar. For more flexibility, use your own
# CMakeLists in combination with the SETUP_MODULE function. The Code for the module is searched in ${CMAKE_CURRENT_SOURCE_DIR}/${_MODULE_NAME}.
# It loads the CMakeLists in the module dir if there is any.
# _MODULE_NAME the name of the module
# Optional second Parameter: list of additional dependencies
# Optional third Parameter: list of style-excludes as regexp.
FUNCTION( ADD_MODULE _MODULE_NAME )
    SET( MODULE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${_MODULE_NAME} )

    # is there a CMakeLists.txt? If yes, use it.
    IF( EXISTS ${MODULE_SOURCE_DIR}/CMakeLists.txt )
        ADD_SUBDIRECTORY( ${_MODULE_NAME} )
        RETURN()
    ENDIF()

    # Optional second parameter: style exclusion list
    SET( _DEPENDENCIES ${ARGV1} )
    SET( _EXCLUDES ${ARGV2} )

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Setup for compilation
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # Let this function do the job. It sets up tests and copies shaders automatically. It additionally configures the stylecheck mechanism for this
    # module.
    SETUP_MODULE( ${_MODULE_NAME}                # name of the module
                 "${MODULE_SOURCE_DIR}"
                 "${_DEPENDENCIES}"
                 "${_EXCLUDES}"
    )
ENDFUNCTION( ADD_MODULE )

# Comfortably setup a module for compilation. This automatically handles the target creation, stylecheck and tests (with fixtures).
# _MODULE_NAME the name of the module
# _MODULE_SOURCE_DIR where to finx the code for the module
# _MODULE_DEPENDENCIES additional dependencies can be added here. This is a list. Use ";" to add multiple additional dependencies
# _MODULE_STYLE_EXCLUDES exclude files from stylecheck matching these regular expressions (list)
FUNCTION( SETUP_MODULE _MODULE_NAME _MODULE_SOURCE_DIR _MODULE_DEPENDENCIES _MODULE_STYLE_EXCLUDES )
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Some common setup
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # setup the target directories and names
    SET( MODULE_NAME ${_MODULE_NAME} )
    SET( MODULE_TARGET_DIR_RELATIVE ${OW_MODULE_DIR_RELATIVE}/${MODULE_NAME} )
    SET( MODULE_TARGET_RESOURCE_DIR_RELATIVE ${OW_SHARE_DIR_RELATIVE}/modules/${MODULE_NAME} )
    SET( MODULE_TARGET_DIR ${PROJECT_BINARY_DIR}/${MODULE_TARGET_DIR_RELATIVE} )
    SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MODULE_TARGET_DIR} )
    SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MODULE_TARGET_DIR} )
    SET( MODULE_SOURCE_DIR ${_MODULE_SOURCE_DIR} )

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Add sources as target
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # Collect the compile-files for this target
    COLLECT_COMPILE_FILES( "${MODULE_SOURCE_DIR}" TARGET_CPP_FILES TARGET_H_FILES TARGET_TEST_FILES )

    # Setup the target
    ADD_LIBRARY( ${MODULE_NAME} SHARED ${TARGET_CPP_FILES} ${TARGET_H_FILES} )
    TARGET_LINK_LIBRARIES( ${MODULE_NAME} ${CMAKE_STANDARD_LIBRARIES} ${OW_LIB_OPENWALNUT} ${Boost_LIBRARIES} ${OPENGL_gl_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES} ${_MODULE_DEPENDENCIES} )

    # Set the version of the library.
    SET_TARGET_PROPERTIES( ${MODULE_NAME} PROPERTIES
        VERSION ${OW_LIB_VERSION}
        SOVERSION ${OW_SOVERSION}
    )

    # Do not forget the install targets
    SETUP_LIB_INSTALL( ${MODULE_NAME} ${MODULE_TARGET_DIR_RELATIVE} "MODULES" )

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Test Setup
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # Setup tests of this target
    SETUP_TESTS( "${TARGET_TEST_FILES}" "${MODULE_NAME}" "${_MODULE_DEPENDENCIES}" )

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Copy Shaders
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    COLLECT_SHADER_FILES( ${MODULE_SOURCE_DIR} TARGET_GLSL_FILES )
    SETUP_SHADERS( "${TARGET_GLSL_FILES}" "${MODULE_TARGET_RESOURCE_DIR_RELATIVE}/shaders" "MODULES" )

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Copy Additional Resources
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    SETUP_RESOURCES_GENERIC( "${MODULE_SOURCE_DIR}/resources" ${MODULE_TARGET_RESOURCE_DIR_RELATIVE} "${_MODULE_NAME}" "MODULES" )

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Style Checker
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # setup the stylechecker. Ignore the platform specific stuff.
    SETUP_STYLECHECKER( "${MODULE_NAME}"
                        "${TARGET_CPP_FILES};${TARGET_H_FILES};${TARGET_TEST_FILES};${TARGET_GLSL_FILES}"  # add all these files to the stylechecker
                        "${_MODULE_STYLE_EXCLUDES}" )                                                      # exlude some ugly files

ENDFUNCTION( SETUP_MODULE )

