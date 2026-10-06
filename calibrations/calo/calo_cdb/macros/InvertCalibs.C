// -- sPHENIX includes
#include <cdbobjects/CDBTTree.h>

#include <Rtypes.h>

// -- c++ includes
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#if defined(__CLING__)
R__LOAD_LIBRARY(libcalo_io.so)
R__LOAD_LIBRARY(libcdbobjects.so)
#endif

// -----------------------------------------------------------------------------
// Helper structure for tracking field inversion statistics
// -----------------------------------------------------------------------------
struct FieldStats
{
  // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
  std::string name;
  std::string type;
  bool inverted{false};
  int totalEntries{0};
  int invertedNonZero{0};
  int zeroOrNaNEntries{0};
  double sumInverted{0.0};
  double minInverted{std::numeric_limits<double>::infinity()};
  double maxInverted{-std::numeric_limits<double>::infinity()};
  // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// -----------------------------------------------------------------------------
// Inversion logic: 1/val if val != 0 else 1
// -----------------------------------------------------------------------------
template <typename T>
T invertValue(T val)
{
  if constexpr (std::is_floating_point_v<T>)
  {
    if (val != static_cast<T>(0) && !std::isnan(val))
    {
      return static_cast<T>(1) / val;
    }
    return static_cast<T>(1);
  }
  else
  {
    if (val != 0)
    {
      return static_cast<T>(1) / val;
    }
    return static_cast<T>(1);
  }
}

// -----------------------------------------------------------------------------
// Helper to generate default output file name: "<stem>_inverted.root"
// -----------------------------------------------------------------------------
inline std::string generateOutputFileName(const std::string &inputFile, const std::string &targetField = "")
{
  std::filesystem::path p(inputFile);
  std::string stem = p.stem().string();
  std::string ext = p.extension().string();
  if (ext.empty())
  {
    ext = ".root";
  }

  std::string suffix = targetField.empty() ? "_inverted" : std::format("_{}_inverted", targetField);
  std::string newFileName = stem + suffix + ext;

  std::filesystem::path parent = p.parent_path();
  if (!parent.empty())
  {
    return (parent / newFileName).string();
  }
  return newFileName;
}

// Case-insensitive field name match helper
inline bool matchesField(const std::string &fieldName, const std::string &target)
{
  if (target.empty())
  {
    return true;
  }
  std::string f = fieldName;
  std::string t = target;
  std::transform(f.begin(), f.end(), f.begin(), ::tolower);
  std::transform(t.begin(), t.end(), t.begin(), ::tolower);
  return f == t;
}

// -----------------------------------------------------------------------------
// Core implementation taking an existing CDBTTree instance
// -----------------------------------------------------------------------------
std::unique_ptr<CDBTTree> InvertCalibs(CDBTTree *in_cdb,
                                      const std::string &outputFile = "",
                                      const std::string &targetField = "")
{
  if (in_cdb == nullptr)
  {
    std::cerr << "Error: Null CDBTTree pointer passed to InvertCalibs.\n";
    return nullptr;
  }

  // Ensure calibrations are loaded
  if (in_cdb->GetFloatEntryMap().empty() && in_cdb->GetDoubleEntryMap().empty() &&
      in_cdb->GetIntEntryMap().empty() && in_cdb->GetUInt64EntryMap().empty() &&
      in_cdb->GetSingleFloatEntryMap().empty() && in_cdb->GetSingleDoubleEntryMap().empty() &&
      in_cdb->GetSingleIntEntryMap().empty() && in_cdb->GetSingleUInt64EntryMap().empty())
  {
    in_cdb->LoadCalibrations();
  }

  const auto &floatMap = in_cdb->GetFloatEntryMap();
  const auto &doubleMap = in_cdb->GetDoubleEntryMap();
  const auto &intMap = in_cdb->GetIntEntryMap();
  const auto &uint64Map = in_cdb->GetUInt64EntryMap();

  const auto &singleFloatMap = in_cdb->GetSingleFloatEntryMap();
  const auto &singleDoubleMap = in_cdb->GetSingleDoubleEntryMap();
  const auto &singleIntMap = in_cdb->GetSingleIntEntryMap();
  const auto &singleUInt64Map = in_cdb->GetSingleUInt64EntryMap();

  bool hasMultiple = !floatMap.empty() || !doubleMap.empty() || !intMap.empty() || !uint64Map.empty();
  bool hasSingle = !singleFloatMap.empty() || !singleDoubleMap.empty() || !singleIntMap.empty() || !singleUInt64Map.empty();

  if (!hasMultiple && !hasSingle)
  {
    std::cerr << "Error: Input CDB tree contains no entries.\n";
    return nullptr;
  }

  // Collect available field names for validation and user feedback
  std::set<std::string> availableFields;
  for (const auto &[_, fmap] : floatMap)
  {
    for (const auto &[pname, _] : fmap)
    {
      availableFields.insert(pname.substr(1));
    }
  }
  for (const auto &[_, dmap] : doubleMap)
  {
    for (const auto &[pname, _] : dmap)
    {
      availableFields.insert(pname.substr(1));
    }
  }
  for (const auto &[_, imap] : intMap)
  {
    for (const auto &[pname, _] : imap)
    {
      availableFields.insert(pname.substr(1));
    }
  }
  for (const auto &[pname, _] : singleFloatMap)
  {
    availableFields.insert(pname.substr(1));
  }
  for (const auto &[pname, _] : singleDoubleMap)
  {
    availableFields.insert(pname.substr(1));
  }
  for (const auto &[pname, _] : singleIntMap)
  {
    availableFields.insert(pname.substr(1));
  }

  // Validate targetField if specified
  if (!targetField.empty())
  {
    bool found = false;
    for (const auto &avail : availableFields)
    {
      if (matchesField(avail, targetField))
      {
        found = true;
        break;
      }
    }
    if (!found)
    {
      std::cerr << std::format("Error: Specified targetField '{}' not found in input CDB tree.\n", targetField);
      std::cerr << "Available fields:\n";
      for (const auto &avail : availableFields)
      {
        std::cerr << "  - " << avail << "\n";
      }
      return nullptr;
    }
  }

  // Prepare output file if specified
  if (!outputFile.empty())
  {
    std::filesystem::path outPath(outputFile);
    if (outPath.has_parent_path())
    {
      std::error_code ec;
      std::filesystem::create_directories(outPath.parent_path(), ec);
      if (ec)
      {
        std::cerr << std::format("Error: Failed to create directory '{}': {}\n",
                                 outPath.parent_path().string(), ec.message());
        return nullptr;
      }
    }
  }

  auto out_cdb = std::make_unique<CDBTTree>(outputFile);
  std::map<std::string, FieldStats> statsMap;

  // Process multiple-entry float values
  for (const auto &[channel, fmap] : floatMap)
  {
    for (const auto &[prefixedName, val] : fmap)
    {
      std::string rawName = prefixedName.substr(1);
      bool shouldInvert = matchesField(rawName, targetField);
      float outVal = shouldInvert ? invertValue(val) : val;
      out_cdb->SetFloatValue(channel, rawName, outVal);

      auto &st = statsMap[rawName];
      st.name = rawName;
      st.type = "float";
      st.inverted = shouldInvert;
      st.totalEntries++;
      if (shouldInvert)
      {
        if (val != 0.0F && !std::isnan(val))
        {
          st.invertedNonZero++;
        }
        else
        {
          st.zeroOrNaNEntries++;
        }
        st.sumInverted += static_cast<double>(outVal);
        st.minInverted = std::min(st.minInverted, static_cast<double>(outVal));
        st.maxInverted = std::max(st.maxInverted, static_cast<double>(outVal));
      }
    }
  }

  // Process multiple-entry double values
  for (const auto &[channel, dmap] : doubleMap)
  {
    for (const auto &[prefixedName, val] : dmap)
    {
      std::string rawName = prefixedName.substr(1);
      bool shouldInvert = matchesField(rawName, targetField);
      double outVal = shouldInvert ? invertValue(val) : val;
      out_cdb->SetDoubleValue(channel, rawName, outVal);

      auto &st = statsMap[rawName];
      st.name = rawName;
      st.type = "double";
      st.inverted = shouldInvert;
      st.totalEntries++;
      if (shouldInvert)
      {
        if (val != 0.0 && !std::isnan(val))
        {
          st.invertedNonZero++;
        }
        else
        {
          st.zeroOrNaNEntries++;
        }
        st.sumInverted += outVal;
        st.minInverted = std::min(st.minInverted, outVal);
        st.maxInverted = std::max(st.maxInverted, outVal);
      }
    }
  }

  // Process multiple-entry int values (copied as-is unless explicitly specified via targetField)
  for (const auto &[channel, imap] : intMap)
  {
    for (const auto &[prefixedName, val] : imap)
    {
      std::string rawName = prefixedName.substr(1);
      bool shouldInvert = !targetField.empty() && matchesField(rawName, targetField);
      int outVal = shouldInvert ? invertValue(val) : val;
      out_cdb->SetIntValue(channel, rawName, outVal);

      auto &st = statsMap[rawName];
      st.name = rawName;
      st.type = "int";
      st.inverted = shouldInvert;
      st.totalEntries++;
      if (shouldInvert)
      {
        if (val != 0)
        {
          st.invertedNonZero++;
        }
        else
        {
          st.zeroOrNaNEntries++;
        }
        st.sumInverted += static_cast<double>(outVal);
        st.minInverted = std::min(st.minInverted, static_cast<double>(outVal));
        st.maxInverted = std::max(st.maxInverted, static_cast<double>(outVal));
      }
    }
  }

  // Process multiple-entry uint64 values (copied as-is)
  for (const auto &[channel, umap] : uint64Map)
  {
    for (const auto &[prefixedName, val] : umap)
    {
      std::string rawName = prefixedName.substr(1);
      out_cdb->SetUInt64Value(channel, rawName, val);
      auto &st = statsMap[rawName];
      st.name = rawName;
      st.type = "uint64";
      st.inverted = false;
      st.totalEntries++;
    }
  }

  // Process single-entry float values
  for (const auto &[prefixedName, val] : singleFloatMap)
  {
    std::string rawName = prefixedName.substr(1);
    bool shouldInvert = matchesField(rawName, targetField);
    float outVal = shouldInvert ? invertValue(val) : val;
    out_cdb->SetSingleFloatValue(rawName, outVal);
  }

  // Process single-entry double values
  for (const auto &[prefixedName, val] : singleDoubleMap)
  {
    std::string rawName = prefixedName.substr(1);
    bool shouldInvert = matchesField(rawName, targetField);
    double outVal = shouldInvert ? invertValue(val) : val;
    out_cdb->SetSingleDoubleValue(rawName, outVal);
  }

  // Process single-entry int values
  for (const auto &[prefixedName, val] : singleIntMap)
  {
    std::string rawName = prefixedName.substr(1);
    bool shouldInvert = !targetField.empty() && matchesField(rawName, targetField);
    int outVal = shouldInvert ? invertValue(val) : val;
    out_cdb->SetSingleIntValue(rawName, outVal);
  }

  // Process single-entry uint64 values
  for (const auto &[prefixedName, val] : singleUInt64Map)
  {
    std::string rawName = prefixedName.substr(1);
    out_cdb->SetSingleUInt64Value(rawName, val);
  }

  // Commit entries
  if (hasMultiple)
  {
    out_cdb->Commit();
  }
  if (hasSingle)
  {
    out_cdb->CommitSingle();
  }

  // Write tree to file if output file specified
  if (!outputFile.empty())
  {
    out_cdb->WriteCDBTTree();
  }

  // Print summary report
  std::cout << "==================================================\n"
            << "InvertCalibs Summary:\n";
  if (!outputFile.empty())
  {
    std::cout << std::format("  Output file: {}\n", outputFile);
  }
  std::cout << "  Fields processed:\n";

  for (const auto &[name, st] : statsMap)
  {
    if (st.inverted)
    {
      double mean = (st.totalEntries > 0) ? (st.sumInverted / st.totalEntries) : 0.0;
      std::cout << std::format("    - {:<18} ({}) -> INVERTED (1/val):\n"
                               "        Total channels:        {}\n"
                               "        Inverted non-zero:     {}\n"
                               "        Zero/NaN (set to 1):   {}\n"
                               "        Min / Max / Mean:      {:.5g} / {:.5g} / {:.5g}\n",
                               st.name, st.type, st.totalEntries,
                               st.invertedNonZero, st.zeroOrNaNEntries,
                               st.minInverted, st.maxInverted, mean);
    }
    else
    {
      std::cout << std::format("    - {:<18} ({}) -> Copied as-is ({} entries)\n",
                               st.name, st.type, st.totalEntries);
    }
  }
  std::cout << "==================================================\n";

  return out_cdb;
}

// -----------------------------------------------------------------------------
// Primary file-based overload
// -----------------------------------------------------------------------------
void InvertCalibs(const std::string &inputFile,
                  const std::string &outputFile = "",
                  const std::string &targetField = "")
{
  if (inputFile.empty())
  {
    std::cerr << "Error: Input file path cannot be empty.\n";
    return;
  }

  if (!std::filesystem::exists(inputFile))
  {
    std::cerr << std::format("Error: Input file '{}' does not exist.\n", inputFile);
    return;
  }

  std::string actualOutputFile = outputFile;
  if (actualOutputFile.empty())
  {
    actualOutputFile = generateOutputFileName(inputFile, targetField);
  }

  std::error_code ec;
  if (std::filesystem::exists(actualOutputFile) && std::filesystem::equivalent(inputFile, actualOutputFile, ec))
  {
    std::cerr << "Error: Output file cannot be identical to the input file.\n";
    return;
  }

  std::cout << std::format("Loading input CDB tree from '{}'...\n", inputFile);
  auto in_cdb = std::make_unique<CDBTTree>(inputFile);
  in_cdb->LoadCalibrations();

  InvertCalibs(in_cdb.get(), actualOutputFile, targetField);
  std::cout << std::format("Successfully created inverted CDB tree: '{}'\n", actualOutputFile);
}

// -----------------------------------------------------------------------------
// Usage guide when called without arguments
// -----------------------------------------------------------------------------
void InvertCalibs()
{
  std::cout << "Usage:\n"
            << "  InvertCalibs(inputFile, [outputFile], [targetField])\n\n"
            << "Arguments:\n"
            << "  inputFile:    Path to the input CDB ROOT file\n"
            << "  outputFile:   (Optional) Path to output ROOT file. Defaults to '<input>_inverted.root'\n"
            << "  targetField:  (Optional) Specific field name to invert. If empty, all float/double fields are inverted.\n\n"
            << "Behavior:\n"
            << "  Inverts calibration values according to: 1/val if val != 0 else 1.\n"
            << "  Non-inverted fields (e.g., status/int fields) are copied to the new tree as-is.\n\n"
            << "Examples:\n"
            << "  InvertCalibs(\"CEMC_ZSCrossCalib.root\")\n"
            << "  InvertCalibs(\"CEMC_ZSCrossCalib.root\", \"CEMC_ZSCrossCalib_inv.root\")\n"
            << "  InvertCalibs(\"HCALIN_HotMap.root\", \"HCALIN_HotMap_inv.root\", \"HCALIN_sigma\")\n";
}
