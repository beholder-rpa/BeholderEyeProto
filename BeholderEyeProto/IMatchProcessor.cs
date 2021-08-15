namespace BeholderEyeProto
{
  using OpenCvSharp;

  public interface IMatchProcessor
  {
    DMatch[][] ProcessAndObtainMatches(Mat queryImage, Mat trainImage, out KeyPoint[] queryKeyPoints, out KeyPoint[] trainKeyPoints);
  }
}