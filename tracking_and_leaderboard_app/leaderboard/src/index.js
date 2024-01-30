    // index.js
    
    import './style';
    import { Component } from 'preact';
    
    export default class App extends Component {
      constructor(props) {
        super(props);
        this.state = {
          computerPick: null,
          result: null,
          leaderboard: [],
        }
	}
    
	componentDidMount() {
		// Fetch data initially
		this.fetchLeaderboardData();
	
		// Fetch data every second
		this.interval = setInterval(() => {
		  this.fetchLeaderboardData();
		}, 1000);
	  }
	
	  componentWillUnmount() {
		// Clear the interval when the component is unmounted
		clearInterval(this.interval);
	  }
	
	  fetchLeaderboardData() {
		fetch('http://localhost:5500/leaderboard')
		  .then(response => response.json())
		  .then(data => {
			console.log(data);
			this.setState({
			  leaderboard: [...data.players],
			});
		  })
		  .catch(error => console.log(error));
	  }

      render() {
        const { leaderboard, computerPick, result } = this.state;
        const sortedLeaderboard = leaderboard.sort((a, b) => b.score > a.score);
        const tableBody = sortedLeaderboard.map((player, index) => (
          <tr>
            <td>{index + 1}</td>
            <td>{player.name}</td>
            <td>{player.score}</td>
          </tr>
        ));
    
        const computerPicked = computerPick ?
          <span class="computer-message">The computer chose {computerPick}</span> : null;
    
        let message;
        if (result !== null) {
          message = result === 1 ?
            <span class="message-content">It's a draw</span> :
            result === 0 ? <span class="message-content fail">You Lost!</span> :
            <span class="message-content success">You won!</span>;
        } else {
          message = null;
        }
    
        return (
          <div class="App">
            <h1>MNIST Leaderboard</h1>
    
            <div class="leaderboard">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Name</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {tableBody}
                </tbody>
              </table>
            </div>
          </div>
        );
      }
    }
