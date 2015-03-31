FILE(REMOVE_RECURSE
  "CMakeFiles/OW_generate_version_header"
  "versionHeader/WToolboxVersion.h"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/OW_generate_version_header.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
