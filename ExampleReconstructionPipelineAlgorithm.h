#pragma once

#include <ImFusion/Base/Algorithm.h>
#include <ImFusion/Core/Parameter.h>
#include <ImFusion/Core/Filesystem/Directory.h>

#include <memory>
#include <string>

namespace ImFusion
{
	class SharedImageSet;

	/// Demonstration of a cone-beam CT reconstruction pipeline.
	/// Chains projection preprocessing, matrix loading, and reconstruction.
	class ExampleReconstructionPipelineAlgorithm : public Algorithm
	{
	public:
		explicit ExampleReconstructionPipelineAlgorithm(SharedImageSet* img);
		~ExampleReconstructionPipelineAlgorithm();

		/// \name Algorithm interface
		///\{
		static bool createCompatible(const DataList& data, Algorithm** a = nullptr);
		void compute() override;
		void configure(const Properties* p) override;
		void configuration(Properties* p) const override;
		OwningDataList takeOutput() override;
		///\}

		Parameter<Filesystem::Path> p_matrixFilePath{"matrixFilePath", "", *this};

		Parameter<Filesystem::Path> p_flatfieldPath{"flatfieldPath", "", *this};

		Parameter<Filesystem::Path> p_darkCurrentPath{"darkCurrentPath", "", *this};

		Parameter<Filesystem::Path> p_deadPixelMaskPath{"deadPixelMaskPath", "", *this};

	private:
		SharedImageSet* m_imgIn = nullptr;                 ///< Input projection images
		std::unique_ptr<SharedImageSet> m_projections;     ///< Preprocessed projections (kept alive for reconstruction)
		std::unique_ptr<SharedImageSet> m_reconstruction;  ///< Reconstruction volume output
	};
}
