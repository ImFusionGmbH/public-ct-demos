#include "ExampleReconstructionPipelineAlgorithm.h"

#include <ImFusion/Base/DataList.h>
#include <ImFusion/Base/OwningDataList.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/CT/ConeBeamMetadata.h>
#include <ImFusion/CT/Legacy/ConeBeamGeometry.h>
#include <ImFusion/CT/ProjectionPreProcessingAlgorithm.h>
#include <ImFusion/CT/ReconstructionAlgorithm.h>
#include <ImFusion/CT/ConvertToConeBeamData.h>
#include <ImFusion/CT/Utils.h>
#include <ImFusion/Core/Properties.h>

#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "CT.ExampleReconstructionPipelineAlgorithm"

namespace ImFusion
{
	ExampleReconstructionPipelineAlgorithm::ExampleReconstructionPipelineAlgorithm(SharedImageSet* imgIn)
		: m_imgIn(imgIn)
	{
	}


	ExampleReconstructionPipelineAlgorithm::~ExampleReconstructionPipelineAlgorithm() = default;


	bool ExampleReconstructionPipelineAlgorithm::createCompatible(const DataList& data, Algorithm** a)
	{
		if (data.size() != 1)
			return false;

		SharedImageSet* img = data.getImage(Data::IMAGESET);
		if (img == nullptr)
			return false;

		if (a)
			*a = new ExampleReconstructionPipelineAlgorithm(img);

		return true;
	}


	void ExampleReconstructionPipelineAlgorithm::configure(const Properties* p)
	{
		Algorithm::configure(p);
	}


	void ExampleReconstructionPipelineAlgorithm::configuration(Properties* p) const
	{
		Algorithm::configuration(p);
	}


	void ExampleReconstructionPipelineAlgorithm::compute()
	{
		m_reconstruction.reset();
		m_projections.reset();
		m_status = Status::Error;

		// Step 1: Preprocess projection images (log conversion, normalization)
		m_imgIn->setModality(Data::XRAY);
		m_projections = CT::Utils::makeConeBeamData();
		auto& metaData = CT::ConeBeamMetadata::get(*m_projections);		

		int border = 0;
		vec4i crop = vec4i::Zero();
		bool flipHor = false;
		bool flipVert = false;	

		for (int i = 0; i < m_imgIn->size(); i++)
		{
			std::shared_ptr<MemImage> mi = nullptr;
			
			mi = m_imgIn->mem(i)->clone();

			if (i == 0 && !CT::Utils::isConeBeamData(*m_imgIn))
			{
				vec3 ext = mi->extent();
				metaData.geometry().detSizeX = ext[0];
				metaData.geometry().detSizeY = ext[1];
			}
			m_projections->add(mi);
			if (m_imgIn->mask(i))
				m_projections->setMask(m_imgIn->mask(i), i);
		}

		if (m_projections->size())
		{
			CT::ProjectionPreProcessingAlgorithm preProc(*m_projections);
			Properties preProcProps;
			preProcProps.setParam("inPlace", true);
			if (!p_deadPixelMaskPath.value().string().empty())
				preProcProps.setParam("deadPixelMaskPath", p_deadPixelMaskPath.value().string());
			if (!p_darkCurrentPath.value().string().empty())
				preProcProps.setParam("darkFieldPath", p_darkCurrentPath.value().string());
			if (!p_flatfieldPath.value().string().empty())
				preProcProps.setParam("flatFieldPath", p_flatfieldPath.value().string());

			// enforce cropping and flipping if required
			//preProcProps.setParam("loadFlipHorizontal", flipHor);
			//preProcProps.setParam("loadFlipVertical", flipVert);
			preProc.configure(&preProcProps);
			preProc.compute();

			if (preProc.status() != Status::Success)
			{
				LOG_ERROR("Projection preprocessing failed with status " << preProc.status());
				return;
			}
		}

		// Step 2: Load projection matrices from ConeBeamGeometry
		CT::ConeBeamGeometry& geom = metaData.geometry();

		if (!p_matrixFilePath.value().string().empty())
		{
			int width = m_projections->get()->width();
			int height = m_projections->get()->height();
			if (!geom.loadMatrices(p_matrixFilePath.value().string(), width, height))
			{
				LOG_ERROR("Failed to load projection matrices from " << p_matrixFilePath.value().string());
				return;
			}
			geom.useMatrices = true;
		}

		// Step 3: Reconstruct volume from preprocessed projections
		CT::ReconstructionAlgorithm recon(*m_projections);
		recon.p_problemMode.setValue("LeastSquaresProblem");
		recon.p_solverMode.setValue("FDK");
		recon.p_regionOfInterestEnabled.setValue(false);
		recon.p_shiftAndScale.setValue(vec2{0.0, 1.0});

		Properties reconProps;

		// FDK solver parameters
		reconProps.setParam("subsetSize", 100);
		reconProps.setParam("filterCacheMaxCost", 0);
		reconProps.setParam("additionalWeights", 0);    // 0=None, 1=Parker, 2=Wang, 3=ParkerAndWang
		reconProps.setParam("normalize", true);

		// Filter parameters
		if (auto* filterProps = reconProps.addSubProperties("Filter"))
		{
			filterProps->setParam("Kernel mode", 2);        // 0=Ram-Lak, 1=Shepp-Logan, 2=Hamming, 3=Parzen
			filterProps->setParam("Scaling", 1.0);
			filterProps->setParam("Alpha", 0.54);           // Hamming window alpha (only used when Kernel mode = Hamming)
			filterProps->setParam("Use Parker weights", false);
			filterProps->setParam("Use Wang weights", false);
		}

		// Volume descriptor parameters
		if (auto* volProps = reconProps.addSubProperties("VolumeDescriptor"))
		{
			volProps->setParam("type", static_cast<int>(PixelType::Float));
			volProps->setParam("width", 256);
			volProps->setParam("height", 256);
			volProps->setParam("slices", 256);
			volProps->setParam("spacing", vec3(0.5, 0.5, 0.5));
			volProps->setParam("center", vec3(0.0, 0.0, 0.0));
		}

		recon.configure(&reconProps);

		recon.compute();

		if (recon.status() != Status::Success)
		{
			LOG_ERROR("Reconstruction failed with status " << recon.status());
			return;
		}

		m_reconstruction = recon.takeOutput().extractFirstImage();
		if (!m_reconstruction)
		{
			LOG_ERROR("No output from reconstruction");
			return;
		}

		m_status = Status::Success;
	}


	OwningDataList ExampleReconstructionPipelineAlgorithm::takeOutput()
	{
		OwningDataList out;
		if (m_projections)
			out.add(std::move(m_projections));
		if (m_reconstruction)
			out.add(std::move(m_reconstruction));
		return out;
	}
}
