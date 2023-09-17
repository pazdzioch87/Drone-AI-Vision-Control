namespace Movement
{
    public interface IRotationChannel
    {
        Task TurnStepLeft();
        Task TurnStepRight();
    }
}
