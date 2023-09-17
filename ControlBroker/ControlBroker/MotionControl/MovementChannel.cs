using LanguageExt;
using Microsoft.AspNetCore.SignalR;
using Movement;

namespace ControlBroker.MotionControl
{
    //public class MovementChannel : IMovementChannel
    //{
    //    public IHubContext<MovementHub, IMovementChannel> Hub { get; }

    //    public MovementChannel(
    //        IHubContext<MovementHub, IMovementChannel> hub)
    //    {
    //        this.Hub = hub;
    //    }

    //    public async Task MoveStepLeft()
    //    {
    //        await this.Hub.Clients.All.MoveStepLeft();
    //        this.LogStep("Left");
    //    }

    //    public async Task MoveStepRight()
    //    {
    //        await this.Hub.Clients.All.MoveStepRight();
    //        this.LogStep("Right");
    //    }

    //    public async Task MoveStepDown()
    //    {
    //        await this.Hub.Clients.All.MoveStepDown();
    //        this.LogStep("Down");
    //    }

    //    public async Task MoveStepUp()
    //    {
    //        await this.Hub.Clients.All.MoveStepUp();
    //        this.LogStep("Up");
    //    }

    //    private void LogStep(string stepDirection) =>
    //        Console.WriteLine(stepDirection);
    //}

    public class MovementHub : Hub<IMovementChannel>
    {
        public async Task MoveStepLeft()
        {
            await this.Clients.All.MoveStepLeft();
            this.LogStep("Left");
        }

        public async Task MoveStepRight()
        {
            await this.Clients.All.MoveStepRight();
            this.LogStep("Right");
        }

        public async Task MoveStepDown()
        {
            await this.Clients.All.MoveStepDown();
            this.LogStep("Down");
        }

        public async Task MoveStepUp()
        {
            await this.Clients.All.MoveStepUp();
            this.LogStep("Up");
        }
        public async Task TakeOff()
        {
            await this.Clients.All.TakeOff();
            this.LogStep("Up");
        }
        public async Task Landing()
        {
            await this.Clients.All.Landing();
            this.LogStep("Up");
        }

        public async Task SendDetection(Detection detection)
        {
            if (detection.Label is null)
            {
                return;
            }
            if (detection.Label.ToLowerInvariant().Contains("down"))
            {
                await this.Clients.All.Landing();
            }
            else if (detection.Label.ToLowerInvariant().Contains("up"))
            {
                await this.Clients.All.TakeOff();
            }
            else
            {
                await this.Clients.All.SendDetection(detection);
            }
            Console.WriteLine($"Detection RECEIVED: {detection.Label} {detection.Confidence}");
        }

        public async Task SendDetections(Detection[] detections)
        {
            var detection = detections.FirstOrDefault();
            if (detection is null || detection?.Label is null)
            {
                return;
            }
            if (detection.Label.ToLowerInvariant().Contains("down"))
            {
                await this.Clients.All.Landing();
            }
            else if (detection.Label.ToLowerInvariant().Contains("up"))
            {
                await this.Clients.All.TakeOff();
            }
            else
            {
                await this.Clients.All.SendDetection(detection);
            }
            Console.WriteLine($"Detection RECEIVED: {detection.Label} {detection.Confidence}");
        }

        public async Task SendDetections2(List<float> coords, string label, string confidence)
        {
            await this.Clients.All.SendDetections2(coords, label, confidence);
        }

        public async Task SendDetections3(string label, string confidence)
        {
            await this.Clients.All.SendDetections3(label, confidence);
            Console.WriteLine($"Detection RECEIVED: {label} {confidence}");
        }

        public async Task Terefere(object some)
        {
            await this.Clients.All.Terefere(some);
        }
        public async Task SendDetectionsEasy(string message)
        {
            await this.Clients.All.SendDetectionsEasy(message);
        }

        private void LogStep(string stepDirection) =>
            Console.WriteLine(stepDirection);
    }
}
