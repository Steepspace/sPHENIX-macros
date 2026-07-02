#include <sepd_eventplanecalib/sEPD_TreeGen.h>

#include <mbd/MbdReco.h>

#include <epd/EpdReco.h>

#include <globalvertex/GlobalVertexReco.h>

#include <centrality/CentralityReco.h>

#include <calotrigger/MinimumBiasClassifier.h>

#include <calostatusskimmer/CaloStatusSkimmer.h>
#include <caloreco/CaloTowerBuilder.h>

#include <ffamodules/CDBInterface.h>
#include <ffamodules/FlagHandler.h>

#include <fun4all/Fun4AllBase.h>
#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllDstOutputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllOutputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/Fun4AllUtils.h>

#include <phool/recoConsts.h>

// root includes --
#include <TSystem.h>

// c++ includes --
#include <fstream>
#include <iostream>
#include <string>


R__LOAD_LIBRARY(libcalo_reco.so)
R__LOAD_LIBRARY(libCaloStatusSkimmer.so)
R__LOAD_LIBRARY(libsepd_eventplanecalib.so)

void Fun4All_sEPD(int nEvents = 0,
                  const std::string& flist_calofit="DST_CALOFITTING.list",
                  const std::string& flist_zdc="DST_ZDC.list",
                  const std::string& flist_sepd="DST_SEPD.list",
                  const std::string& output = "test.root",
                  const std::string& output_tree = "tree.root",
                  const std::string& dbtag = "newcdbtag")
{
  std::cout << "########################" << std::endl;
  std::cout << "Run Parameters" << std::endl;
  std::cout << "input calofit: " << flist_calofit << std::endl;
  std::cout << "input zdc: " << flist_zdc << std::endl;
  std::cout << "input sepd: " << flist_sepd << std::endl;
  std::cout << "output: " << output << std::endl;
  std::cout << "output tree: " << output_tree << std::endl;
  std::cout << "nEvents: " << nEvents << std::endl;
  std::cout << "dbtag: " << dbtag << std::endl;
  std::cout << "########################" << std::endl;

  Fun4AllServer* se = Fun4AllServer::instance();


  // Extract runnumber from first file within list
  int runnumber;
  bool isFileList = true;
  // single file
  if (flist_calofit.ends_with(".root"))
  {
    std::pair<int, int> runseg = Fun4AllUtils::GetRunSegment(flist_calofit);
    runnumber = runseg.first;
    isFileList = false;
  }
  // list of files
  else
  {
    std::ifstream infile_stream(flist_calofit);
    if (!infile_stream) {
      std::cout << "Error: Could not open file list " << flist_calofit << std::endl;
      return;
    }
    std::string filepath;
    getline(infile_stream, filepath);
    std::pair<int, int> runseg = Fun4AllUtils::GetRunSegment(filepath);
    runnumber = runseg.first;
    infile_stream.close();
  }

  recoConsts* rc = recoConsts::instance();

  // conditions DB flags and timestamp
  rc->set_StringFlag("CDB_GLOBALTAG", dbtag);
  rc->set_uint64Flag("TIMESTAMP", runnumber);
  CDBInterface::instance()->Verbosity(Fun4AllBase::VERBOSITY_SOME);

  SubsysReco* flag = new FlagHandler();
  se->registerSubsystem(flag);

  // Remove incomplete events from event combiner
  CaloStatusSkimmer* css = new CaloStatusSkimmer("CaloStatusSkimmer");
  se->registerSubsystem(css);

  // MBD Reconstruction
  SubsysReco* mbdreco = new MbdReco();
  se->registerSubsystem(mbdreco);

  CaloTowerDefs::BuilderType buildertype = CaloTowerDefs::kPRDFTowerv4;

  // sEPD Reconstruction--Calib Info: Packets -> TOWERS_SEPD
  CaloTowerBuilder *caEPD = new CaloTowerBuilder("SEPDBUILDER");
  caEPD->set_detector_type(CaloTowerDefs::SEPD);
  caEPD->set_builder_type(buildertype);
  caEPD->set_processing_type(CaloWaveformProcessing::TEMPLATE);
  caEPD->set_nsamples(12);
  caEPD->set_offlineflag();
  se->registerSubsystem(caEPD);

  // sEPD Reconstruction--Calib Info
  SubsysReco* epdreco = new EpdReco();
  se->registerSubsystem(epdreco);

  // Official vertex storage
  SubsysReco* gvertex = new GlobalVertexReco();
  se->registerSubsystem(gvertex);

  // Minimum Bias Classifier
  MinimumBiasClassifier* mb = new MinimumBiasClassifier();
  mb->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->registerSubsystem(mb);

  // Centrality
  CentralityReco* cent = new CentralityReco();
  se->registerSubsystem(cent);

  // sEPD Tree Gen
  sEPD_TreeGen* sepd_gen = new sEPD_TreeGen();
  sepd_gen->Verbosity(1);
  se->registerSubsystem(sepd_gen);

  Fun4AllInputManager* In = new Fun4AllDstInputManager("calofitting");
  if (isFileList)
  {
    In->AddListFile(flist_calofit);
  }
  else
  {
    In->AddFile(flist_calofit);
  }
  se->registerInputManager(In);

  Fun4AllInputManager* In2 = new Fun4AllDstInputManager("zdc");
  if (isFileList)
  {
    In2->AddListFile(flist_zdc);
  }
  else
  {
    In2->AddFile(flist_zdc);
  }
  se->registerInputManager(In2);

  Fun4AllInputManager* In3 = new Fun4AllDstInputManager("sepd");
  if (isFileList)
  {
    In3->AddListFile(flist_sepd);
  }
  else
  {
    In3->AddFile(flist_sepd);
  }
  se->registerInputManager(In3);

  Fun4AllOutputManager* out = new Fun4AllDstOutputManager("dstout", output_tree);
  out->SplitLevel(99); // so we can look at its content from the root prompt
  out->AddNode("EventPlaneData");
  se->registerOutputManager(out);

  se->Verbosity(Fun4AllBase::VERBOSITY_QUIET);
  se->run(nEvents);
  se->End();

  se->dumpHistos(output);

  CDBInterface::instance()->Print();  // print used DB files
  se->PrintTimer();
  delete se;
  std::cout << "All done!" << std::endl;
  gSystem->Exit(0);
}
