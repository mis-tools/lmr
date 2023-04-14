#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkPadImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkTimeProbe.h"

 
 
void Usage(char * argv)
{
    std::cout << "Usage: " << argv << " -ref targetimagefile -flo labelmapfile  [-aff xfmtransformfile] [-res outputfile]" << std::endl;
}
 
int main(int argc, char * argv[])
{
    // Usage:
    // LabelMapResampler <labelmap> <targetimage>
    bool referenceImageFlag = false;
    bool floatingImageFlag = false;
    bool affineMatrixFlag = false;
    bool outputResultFlag = false;

    char * floatingImageName;
    char * referenceImageName;
    char * affineMatrixName;
    char * outputResultName;
    // Parse input as in reg_resample
    /* read the input parameter */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
           strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
           strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return EXIT_FAILURE;
        }
        else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) ||
                (strcmp(argv[i],"--ref")==0)){
            referenceImageName=argv[++i];
            referenceImageFlag=1;
        }
        else if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) ||
                (strcmp(argv[i],"--flo")==0)){
            floatingImageName=argv[++i];
            floatingImageFlag=1;
        }
        else if(strcmp(argv[i], "-aff") == 0 ||
                (strcmp(argv[i],"--aff")==0)){
            affineMatrixName=argv[++i];
            affineMatrixFlag=1;
        }
        else if((strcmp(argv[i],"-res")==0) || (strcmp(argv[i],"-result")==0) ||
                (strcmp(argv[i],"--res")==0)){
                outputResultName=argv[++i];
                outputResultFlag=1;
        }
        else{
            fprintf(stderr,"[LabelMapResampler] Err:\tParameter %s unknown.\n",argv[i]);
            Usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if ( ! referenceImageFlag || ! floatingImageFlag )
    {
        fprintf(stderr,"[LabelMapResampler] Err:\t Target and floating image must be specified.\n");
        return EXIT_FAILURE;
    }



    typedef unsigned short PixelType;
    typedef itk::Image<PixelType, 3> ImageType;
    typedef itk::Image<unsigned char, 3> UCharImageType;
    typedef itk::Image<float, 3> FloatImageType;



    // Validate input label image information
    {
        itk::ImageIOBase::Pointer imageIO;
        try
        {
            imageIO = itk::ImageIOFactory::CreateImageIO(floatingImageName, itk::ImageIOFactory::ReadMode);
        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "[LabelMapResampler] Error reading input image" << std::endl;
            return EXIT_FAILURE;
        }
        imageIO->SetFileName(floatingImageName);
        imageIO->ReadImageInformation();
        if ( imageIO->GetNumberOfDimensions() != 3 )
        {
            std::cerr << "[LabelMapResampler] Error: Wrong label image dimension. Only three-dimensional images are supported." << std::endl;
            return EXIT_FAILURE;
        }
        if ( imageIO->GetPixelType() != imageIO->GetPixelTypeFromString("scalar") )
        {
            std::cerr << "[LabelMapResampler] Error: Wrong label image type. Only scalar images are supported." << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "[LabelMapResampler] Floating image origin [ " << imageIO->GetOrigin(0) << ", " << imageIO->GetOrigin(1) << ", " << imageIO->GetOrigin(2) << " ] " << std::endl;
        std::cout << "[LabelMapResampler] Floating image size [ " << imageIO->GetDimensions(0)  << ", " << imageIO->GetDimensions(1) << ", " << imageIO->GetDimensions(2) << " ] " << std::endl;
    }


    // Read input labelmap
    ImageType::Pointer labelmap;
    typedef itk::ImageFileReader<ImageType> ReaderType;


    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( floatingImageName );
    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject& kExcp)
    {
        std::cerr << kExcp << std::endl;
        return EXIT_FAILURE;
    }
    labelmap = reader->GetOutput();

    // Get all label values
    std::vector<PixelType> labelvalues;
    {
        std::set<PixelType> labelvaluesset;
        // Loop over the labelmap
        for ( int i = 0; i < labelmap->GetBufferedRegion().GetNumberOfPixels() ; i++ )
        {
            labelvaluesset.insert(labelmap->GetBufferPointer()[i]);
        }
        // Insert the labels values in a vector container
        for ( std::set<PixelType>::iterator it = labelvaluesset.begin(); it != labelvaluesset.end() ; ++it )
        {
            labelvalues.push_back(*it);
        }
        std::cout << "[LabelMapResampler] Found "  << labelvalues.size() << " labels in labelmap" << std::endl;
    }
    if ( labelvalues.size() > 255 )
    {
        std::cerr << "[LabelMapResampler] Error: Too many labels in labelmap" << std::endl;
        return EXIT_FAILURE;
    }


    // Read target image dimensions
    ImageType::SizeType outputSize;
    ImageType::SpacingType outputSpacing;
    ImageType::DirectionType outputDirection;
    double outputOrigin[3];
    double originalOrigin[3];
    ImageType::SizeType originalSize;
    int croppedCornerVoxelInOriginalImage[3];
    {
        itk::ImageIOBase::Pointer imageIO;
        try
        {
                imageIO = itk::ImageIOFactory::CreateImageIO(referenceImageName, itk::ImageIOFactory::ReadMode);
        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "Error reading target image" << std::endl;
            return EXIT_FAILURE;
        }
        imageIO->SetFileName(referenceImageName);
        imageIO->ReadImageInformation();

        std::cout << "[LabelMapResampler] Target image origin [ " << imageIO->GetOrigin(0) << ", " << imageIO->GetOrigin(1) << ", " << imageIO->GetOrigin(2) << " ] " << std::endl;
        std::cout << "[LabelMapResampler] Target image size [ " << imageIO->GetDimensions(0)  << ", " << imageIO->GetDimensions(1) << ", " << imageIO->GetDimensions(2) << " ] " << std::endl;
        for ( unsigned int i = 0; i < 3; i++ )
        {
            outputOrigin[i] = imageIO->GetOrigin(i);
            outputSize[i] = imageIO->GetDimensions(i);
            outputSpacing[i] = imageIO->GetSpacing(i);
        }
#ifdef DEBUG        
        std::cout << "[LabelMapResampler DEBUG] Direction " << std::endl;
#endif 
        for (unsigned int i = 0; i < 3; i++ )
        {
            std::vector<double> direction = imageIO->GetDirection(i);
            outputDirection(0,i) = direction[0];
            outputDirection(1,i) = direction[1];
            outputDirection(2,i) = direction[2];
#ifdef DEBUG        
            std::cout << "[LabelMapResampler DEBUG] " << direction[0] << " " << direction[1] << " " << direction[2] << std::endl;
#endif 
        }
    }
    


    //Transform
    typedef itk::AffineTransform<double, 3> AffineTransformType;
    AffineTransformType::Pointer transform = AffineTransformType::New();
    if ( ! affineMatrixFlag )
    {
        transform->SetIdentity();
    }
    else
    {
        itk::Matrix<double, 3, 3> transformrotation;
        itk::Vector<double, 3> transformtranslation;
        char buffer[256];
        FILE * fh = fopen(affineMatrixName, "r");
        int row = 0;
        while ( ! feof(fh) && row < 3)
        {
            char * c = fgets(buffer, 256, fh);
            float rowdata[4];
            if (sscanf(buffer, "%e %e %e %e", &rowdata[0], &rowdata[1], &rowdata[2], &rowdata[3]))
            {
                for ( int i = 0; i < 3; i++ )
                {
                    transformrotation(row, i) = rowdata[i];
                }
                transformtranslation[row] = rowdata[3];
                row++;
            }
        }
        if ( row != 3 )
        {
            std::cerr << "[LabelMapResampler] Error: invalid transform file." << std::endl;
            return EXIT_FAILURE;
        }
        // Code from convert3d to generate itk transform from xfm matrix.
        vnl_vector<double> v_lps_to_ras(3, 1.0);
        v_lps_to_ras[0] = v_lps_to_ras[1] = -1.0;
        vnl_diag_matrix<double> m_lps_to_ras(v_lps_to_ras);
        vnl_matrix<double> oldmatrix = transformrotation.GetVnlMatrix();
        transformrotation.GetVnlMatrix().update(m_lps_to_ras * oldmatrix * m_lps_to_ras);
        transformtranslation.GetVnlVector().update(m_lps_to_ras * transformtranslation.GetVnlVector());
        transform->SetMatrix(transformrotation);
        transform->SetOffset(transformtranslation);
        
    }
#ifdef DEBUG    
    std::cout << "[LabelMapResampler] XFM matrix: "<< transform->GetMatrix() << std::endl;
    std::cout << "[LabelMapResampler] XFM Translation"<< transform->GetTranslation() << std::endl;
#endif    



    ImageType::SizeType labelmapSize = labelmap->GetLargestPossibleRegion().GetSize();
 
    std::cout << "[LabelMapResampler] Labelmap size " << labelmapSize << std::endl;
    // Find bounding box of the resampled combined labels
    // In the future, we should only resample to the cropped size.
    // At the moment the code below only reports the cropped size.
    bool resampletocropped = true;
    if ( resampletocropped )
    {
        
        typedef itk::BinaryThresholdImageFilter <ImageType, UCharImageType> BinaryThresholdImageFilterType;
        BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
        thresholdFilter->SetInput(labelmap);
        thresholdFilter->SetLowerThreshold(1);
        thresholdFilter->SetInsideValue(1);
        thresholdFilter->SetOutsideValue(0);

        thresholdFilter->Update();

        typedef itk::NearestNeighborInterpolateImageFunction<UCharImageType, double> InterpolatorType;
        InterpolatorType::Pointer interpolator = InterpolatorType::New();


        typedef itk::ResampleImageFilter<UCharImageType, UCharImageType> ResampleImageFilterType;
        ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();

        resample->SetDefaultPixelValue(0);
        resample->SetInput(thresholdFilter->GetOutput());
        resample->SetInterpolator(interpolator);

        resample->SetSize(outputSize);
        resample->SetOutputSpacing(outputSpacing);
        resample->SetTransform(transform);
        resample->SetOutputOrigin(outputOrigin);
        resample->SetOutputDirection(outputDirection);
        resample->UpdateLargestPossibleRegion();


        typedef itk::ImageMaskSpatialObject< 3 > ImageMaskSpatialObjectType;

        ImageMaskSpatialObjectType::Pointer imagemask = ImageMaskSpatialObjectType::New();

        imagemask->SetImage ( resample->GetOutput() );
        imagemask->Update();
        typedef itk::ImageRegion<3> RegionType;
        RegionType region = imagemask->GetAxisAlignedBoundingBoxRegion();

        std::cout << "[LabelMapResampler] Cropped size in target space: " << region.GetSize() << std::endl;
        // Save the original size and origin
        for ( int i = 0; i < 3; i++ )
        {
            originalSize[i] = outputSize[i];
            originalOrigin[i] = outputOrigin[i];
            outputSize[i] = region.GetSize(i);
#ifdef DEBUG
            std::cout << "[DEBUG LabelMapResampler] Output size " << outputSize[i] << std::endl;
#endif        
        }

        // Calculate new origin
        vnl_matrix<double> voxeltoworld(4,4, 0.0);
        voxeltoworld.update(outputDirection.GetVnlMatrix(), 0, 0);
        voxeltoworld(0, 3) = outputOrigin[0];
        voxeltoworld(1, 3) = outputOrigin[1];
        voxeltoworld(2, 3) = outputOrigin[2];
        voxeltoworld(3, 3) = 1.0;
#ifdef DEBUG
        std::cout << "[DEBUG LabelMapResampler] Voxel to world " << std::endl  << voxeltoworld << std::endl;
#endif        
        vnl_vector<double> cornervoxel(4);
        cornervoxel[0] = region.GetIndex(0) * outputSpacing[0];
        cornervoxel[1] = region.GetIndex(1) * outputSpacing[1];
        cornervoxel[2] = region.GetIndex(2) * outputSpacing[2];
        cornervoxel[3] = 1.0;

        vnl_vector<double> neworigin(4);
        neworigin = voxeltoworld * cornervoxel;
#ifdef DEBUG        
        std::cout << "[DEBUG LabelMapResampler] New origin: " << neworigin << std::endl;
#endif        


        outputOrigin[0] = neworigin[0];
        outputOrigin[1] = neworigin[1];
        outputOrigin[2] = neworigin[2];
        // Save corner voxel for padding at the end
        croppedCornerVoxelInOriginalImage[0] = region.GetIndex(0);
        croppedCornerVoxelInOriginalImage[1] = region.GetIndex(1);
        croppedCornerVoxelInOriginalImage[2] = region.GetIndex(2);
    }

    // Create an output image for the combined labelmap
    ImageType::Pointer combinedlabels = ImageType::New();

    combinedlabels->SetRegions(outputSize);
    combinedlabels->Allocate();
    combinedlabels->FillBuffer(0);
    combinedlabels->SetDirection(outputDirection);
    combinedlabels->SetSpacing(outputSpacing);
    combinedlabels->SetOrigin(outputOrigin);

    // Create a probability image that holds the current probability
    UCharImageType::Pointer probabilityimage = UCharImageType::New();

    probabilityimage->SetRegions(outputSize);
    probabilityimage->Allocate();
    probabilityimage->FillBuffer(0);
    probabilityimage->SetDirection(outputDirection);
    probabilityimage->SetSpacing(outputSpacing);
    probabilityimage->SetOrigin(outputOrigin);


    typedef  itk::ImageFileWriter<ImageType> WriterType;
    typedef  itk::ImageFileWriter<UCharImageType> UCharWriterType;
    typedef  itk::ImageFileWriter<FloatImageType> FloatWriterType;

    // 
    printf("[LabelMapResampler] %8s %13s %13s %8s\n", "LabelID", "Input BB", "Output BB", "Timing");
    // Split the labelmap into individual binary masks
    // The first labelvalue is assumed to be the background
    //#pragma omp parallel for schedule(dynamic)
    for ( int i = 0; i < labelvalues.size(); i++ )
    {
        itk::TimeProbe clock;
        clock.Start();
        PixelType labelvalue = labelvalues[i];

        // Create a binary mask by thresholding
        typedef itk::BinaryThresholdImageFilter <ImageType, UCharImageType> BinaryThresholdImageFilterType;
        //typedef itk::BinaryThresholdImageFilter <ImageType, FloatImageType> BinaryThresholdImageFilterType;
        BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
        thresholdFilter->SetInput(labelmap);
        thresholdFilter->SetLowerThreshold(labelvalue);
        thresholdFilter->SetUpperThreshold(labelvalue);
        thresholdFilter->SetInsideValue(255);
        thresholdFilter->SetOutsideValue(0);

        thresholdFilter->Update();

        typedef itk::ImageMaskSpatialObject< 3 > ImageMaskSpatialObjectType;

        ImageMaskSpatialObjectType::Pointer labelimagemask = ImageMaskSpatialObjectType::New();

        labelimagemask->SetImage ( thresholdFilter->GetOutput() );
        labelimagemask->Update();
        typedef itk::ImageRegion<3> RegionType;
        RegionType labelregion = labelimagemask->GetAxisAlignedBoundingBoxRegion();

        //std::cout << "Binary label cropped size " << labelregion.GetSize() << " , index :"<< labelregion.GetIndex()  <<std::endl;

        // Pad to make space for fuzzy mask
        itk::Index<3> paddedorigin;
        itk::Index<3> paddedcorner;
        for ( int j = 0; j < 3; j++ )
        {
            paddedorigin[j] = labelregion.GetIndex()[j] - 3;
            paddedcorner[j] = labelregion.GetUpperIndex()[j] + 3;
            if ( paddedorigin[j] < 0 )
                  paddedorigin[j] = 0;
            if ( paddedcorner[j] >= labelmapSize[j] )
                  paddedcorner[j] = labelmapSize[j] - 1;
        }
        labelregion.SetIndex(paddedorigin);
        labelregion.SetUpperIndex(paddedcorner);

        //std::cout << "Binary label padded to size " << labelregion.GetSize() << " , index :"<< labelregion.GetIndex()  << std::endl;

        typedef itk::ExtractImageFilter< UCharImageType, UCharImageType> ExtractImageFilterType;
        ExtractImageFilterType::Pointer labelextractfilter = ExtractImageFilterType::New();
        labelextractfilter->SetInput(thresholdFilter->GetOutput());
        labelextractfilter->SetExtractionRegion(labelregion);


        // Smooth the binary mask to create a fuzzy mask
        //std::cout << "Smoothing input label " << labelvalue << std::endl;
        //typedef itk::SmoothingRecursiveGaussianImageFilter<UCharImageType, UCharImageType>  GaussianImageFilterType;
        //typedef itk::SmoothingRecursiveGaussianImageFilter<FloatImageType, FloatImageType>  GaussianImageFilterType;
        // Alternatively, use the DiscreteGaussianImageFilter
        typedef itk::DiscreteGaussianImageFilter< UCharImageType, UCharImageType>  GaussianImageFilterType;
        //typedef itk::DiscreteGaussianImageFilter< FloatImageType, FloatImageType>  GaussianImageFilterType;
        
        GaussianImageFilterType::Pointer gaussfilter = GaussianImageFilterType::New();
        //gaussfilter->SetInput(thresholdFilter->GetOutput());
        gaussfilter->SetInput(labelextractfilter->GetOutput());
        //gaussfilter->SetSigma(1.0); // In physical coordinates as default
        gaussfilter->SetVariance(1.0 * 1.0);
        gaussfilter->SetUseImageSpacingOn();
        gaussfilter->Update();


        // Crop the fuzzy mask to a tight bounding box
        // First find the bounding box region
        //typedef itk::ImageMaskSpatialObject< 3 > ImageMaskSpatialObjectType;

        ImageMaskSpatialObjectType::Pointer imagemask = ImageMaskSpatialObjectType::New();

        imagemask->SetImage ( gaussfilter->GetOutput() );
        imagemask->Update();
        itk::Size<3> size;
        itk::Index<2> index;
        typedef itk::ImageRegion<3> RegionType;
        RegionType region = imagemask->GetAxisAlignedBoundingBoxRegion();

        //std::cout << "Fuzzy label cropped size " << region.GetSize() << std::endl;

        // Extract the cropped image using the bounding box region
        //typedef itk::ExtractImageFilter< UCharImageType, UCharImageType> ExtractImageFilterType;
        //ExtractImageFilterType::Pointer extractfilter = ExtractImageFilterType::New();
        //extractfilter->SetInput(gaussfilter->GetOutput());
        //extractfilter->SetExtractionRegion(region);


        // Resample the cropped fussy mask image to the target size
        //std::cout << "Resampling  " << std::endl;


        typedef itk::ResampleImageFilter<UCharImageType, UCharImageType> ResampleImageFilterType;
        ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();

      
        // Default interpolator is linear. Comment in the three lines below in
        // order to use a nearest interpolator.
        //typedef itk::NearestNeighborInterpolateImageFunction<UCharImageType, double> InterpolatorType;
        //InterpolatorType::Pointer interpolator = InterpolatorType::New();
        //resample->SetInterpolator(interpolator);

        resample->SetDefaultPixelValue(0);
        resample->SetInput(gaussfilter->GetOutput());
        //resample->SetNumberOfThreads(1);
        //resample->SetInput(extractfilter->GetOutput());

        resample->SetSize(outputSize);
        resample->SetOutputSpacing(outputSpacing);
        resample->SetTransform(transform);
        resample->SetOutputOrigin(outputOrigin);
        resample->SetOutputDirection(outputDirection);
        resample->UpdateLargestPossibleRegion();

        UCharImageType::Pointer output = resample->GetOutput();

#ifdef DEBUG        
        UCharWriterType::Pointer maskwriter = UCharWriterType::New();
        char maskfilename[64];
        sprintf(maskfilename, "mask_%03d.nii.gz", labelvalues[i] );
        maskwriter->SetFileName(maskfilename);
        maskwriter->SetInput(output);
        maskwriter->Update();
#endif        

        // Get the region of interest in the resampled image
        ImageMaskSpatialObjectType::Pointer imagemaskresampled = ImageMaskSpatialObjectType::New();

        imagemaskresampled->SetImage ( resample->GetOutput() );
        imagemaskresampled->Update();
        RegionType regionresampled = imagemaskresampled->GetAxisAlignedBoundingBoxRegion();


        // Iterate over the region in resampled space. Update the probability
        // and label value if the mask probability value is greater than the
        // previously set probability.
        typedef itk::ImageRegionIterator<UCharImageType> UCharImageIteratorType;
        typedef itk::ImageRegionConstIterator<UCharImageType> UCharConstImageIteratorType;
        typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;

        UCharConstImageIteratorType regioniterator_singlelabel(resample->GetOutput(), regionresampled);
        ImageIteratorType regioniterator_combinedlabels(combinedlabels, regionresampled);
        UCharImageIteratorType regioniterator_probability(probabilityimage, regionresampled);

        regioniterator_singlelabel.GoToBegin();
        regioniterator_combinedlabels.GoToBegin();
        regioniterator_probability.GoToBegin();

        while ( ! regioniterator_singlelabel.IsAtEnd() )
        {
            // Use the lowest labelvalue for equal voting to ensure
            // deterministic result for multithreaded usage
            if ( regioniterator_singlelabel.Get() > regioniterator_probability.Get() )
            {
                // Update combined label image with the current label.
                regioniterator_combinedlabels.Set(labelvalues[i]);
                // Update probability image with the current probability.
                regioniterator_probability.Set(regioniterator_singlelabel.Get());
            }
            ++regioniterator_singlelabel;
            ++regioniterator_combinedlabels;
            ++regioniterator_probability;
        }
        clock.Stop();
        printf("[LabelMapResampler] %8d [%3d %3d %3d] [%3d %3d %3d] %f\n", labelvalues[i], 
                                                                        int(region.GetSize(0)), 
                                                                        int(region.GetSize(1)), 
                                                                        int(region.GetSize(2)), 
                                                                        int(regionresampled.GetSize(0)),
                                                                        int(regionresampled.GetSize(1)),
                                                                        int(regionresampled.GetSize(2)),
                                                                        clock.GetTotal()
                                                                        );


    }

    // Write combined labels image
    std::cout << "[LabelMapResampler] Writing combined labels ... " << std::endl;
    WriterType::Pointer outputWriter = WriterType::New();
    if ( outputResultFlag )
    {
        outputWriter->SetFileName(outputResultName);
    }
    else
    {
        outputWriter->SetFileName("resampled_labels.nii.gz");
    }

    if (resampletocropped)
    {
        // Pad the cropped output image so that is has the same dimensions
        // and origin as the target image
        ImageType::SizeType lowerExtendRegion;
        ImageType::SizeType upperExtendRegion;
        for ( int i = 0; i < 3; i++ )
        {
            upperExtendRegion[i] = originalSize[i] - (croppedCornerVoxelInOriginalImage[i] + outputSize[i]);
            lowerExtendRegion[i] = croppedCornerVoxelInOriginalImage[i];
        }

        typedef itk::ConstantPadImageFilter<ImageType, ImageType> PadImageFilterType;
        PadImageFilterType::Pointer padfilter = PadImageFilterType::New();
        padfilter->SetInput(combinedlabels);
        padfilter->SetPadLowerBound(lowerExtendRegion);
        padfilter->SetPadUpperBound(upperExtendRegion);
        padfilter->SetConstant(0);
        padfilter->Update();

        outputWriter->SetInput(padfilter->GetOutput());
    }
    else
    {
        outputWriter->SetInput(combinedlabels);
    }
    outputWriter->Update();

    /*
     // Write probability image
     {
        std::cout << "Writing probability image ... " << std::endl;
        UCharWriterType::Pointer outputWriter = UCharWriterType::New();
        outputWriter->SetFileName("probability_image.nii.gz");
        outputWriter->SetInput(probabilityimage);
        outputWriter->Update();
     }
     */
  return EXIT_SUCCESS;
}
