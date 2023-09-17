using LanguageExt.Pipes;

namespace Movement
{
    public class Detection
    {
        public List<float>? Coords { get; set; }
        public string? Label { get; set; }
        public float Confidence { get; set; }
    }

    public interface IMovementChannel
    {
        Task MoveStepUp();
        Task MoveStepDown();
        Task MoveStepLeft();
        Task MoveStepRight();
        Task TakeOff();
        Task Landing();
        Task SendDetection(Detection detection);
        Task SendDetections(Detection[] detections);
        Task SendDetections2(List<float> coords, string label, string confidence);
        Task SendDetections3(string label, string confidence);
        Task SendDetectionsEasy(string message);
        Task Terefere(object some);
    }
}