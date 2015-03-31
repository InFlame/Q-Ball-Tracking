FILE(REMOVE_RECURSE
  "CMakeFiles/stylecheck"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/stylecheck.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
