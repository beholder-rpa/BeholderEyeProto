namespace BeholderEyeProto
{
  using OpenCvSharp;
  using System.Collections.Generic;

  public interface IMatchMaskFactory
  {
    Mat CreateMatchMask(DMatch[][] matches, KeyPoint[] queryKeyPoints, KeyPoint[] trainKeyPoints, out IList<DMatch> goodMatches);
  }
}