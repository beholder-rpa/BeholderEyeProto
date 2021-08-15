namespace BeholderEyeProto
{
  using Microsoft.Extensions.Logging;
  using OpenCvSharp;
  using System.Collections.Generic;

  class Program
  {
    static ILoggerFactory _loggerFactory;

    static void Main()
    {
      _loggerFactory = LoggerFactory.Create(builder =>
      {
        builder
            .AddFilter("Microsoft", LogLevel.Warning)
            .AddFilter("System", LogLevel.Warning)
            .AddConsole();
      });

      ObtainCorrespondingImageLocations("./images/goldCookie.png", "./images/cookiescreen.png", "foo.png");
      ObtainCorrespondingImageLocations("./images/goldCookie.png", "./images/cookiescreen-1.png", "foo-1.png");
    }

    static IEnumerable<IEnumerable<Point>> ObtainCorrespondingImageLocations(string queryImagePath, string trainImagePath, string outputImagePath = null)
    {
      using var queryImage = Cv2.ImRead(queryImagePath);
      using var trainImage = Cv2.ImRead(trainImagePath);

      var locationsResult = new List<IEnumerable<Point>>();

      var matchProcessor = new SiftFlannMatchProcessor();
      DMatch[][] knn_matches = matchProcessor.ProcessAndObtainMatches(queryImage, trainImage, out KeyPoint[] queryKeyPoints, out KeyPoint[] trainKeyPoints);

      var matchMaskFactory = new MatchMaskFactory(_loggerFactory.CreateLogger<MatchMaskFactory>());
      using var mask = matchMaskFactory.CreateMatchMask(knn_matches, queryKeyPoints, trainKeyPoints, out var goodMatches);

      if (goodMatches.Count > 4)
      {
        // Use Homeography to obtain a perspective-corrected rectangle of the target in the query image.
        var sourcePoints = new Point2f[goodMatches.Count];
        var destinationPoints = new Point2f[goodMatches.Count];
        for (int i = 0; i < goodMatches.Count; i++)
        {
          DMatch match = goodMatches[i];
          sourcePoints[i] = queryKeyPoints[match.QueryIdx].Pt;
          destinationPoints[i] = trainKeyPoints[match.TrainIdx].Pt;
        }

        Point[] targetPoints;
        using var homography = Cv2.FindHomography(InputArray.Create(sourcePoints), InputArray.Create(destinationPoints), HomographyMethods.Ransac, 5.0);
        {
          Point2f[] obj_corners = {
          new Point2f(0, 0),
          new Point2f(queryImage.Cols, 0),
          new Point2f(queryImage.Cols, queryImage.Rows),
          new Point2f(0, queryImage.Rows)
        };

          Point2f[] dest = Cv2.PerspectiveTransform(obj_corners, homography);
          targetPoints = new Point[dest.Length];
          for (int i = 0; i < dest.Length; i++)
          {
            targetPoints[i] = dest[i].ToPoint();
          }
        }

        locationsResult.Add(targetPoints);
      }

      if (!string.IsNullOrWhiteSpace(outputImagePath))
      {
        byte[] maskBytes = new byte[mask.Rows * mask.Cols];
        Cv2.Polylines(trainImage, locationsResult, true, new Scalar(255, 0, 0), 3, LineTypes.AntiAlias);
        using var outImg = new Mat();
        Cv2.DrawMatches(queryImage, queryKeyPoints, trainImage, trainKeyPoints, goodMatches, outImg, new Scalar(0, 255, 0), flags: DrawMatchesFlags.NotDrawSinglePoints);
        outImg.SaveImage(outputImagePath);
      }

      return locationsResult;
    }
  }
}