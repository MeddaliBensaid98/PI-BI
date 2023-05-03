import { Component } from '@angular/core';
import { AuthService } from '../../services/auth.service';
import { OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-update',
  templateUrl: './update.component.html',
  styleUrls: ['./update.component.css'],
})
export class UpdateComponent implements OnInit {
  user: any;
  username: string;
  password: string;
  cin: number;
  userByUsername: any;
  tel: number;
  dateBirth: Date;
  email: string;
  nom: string;
  surname: string;
  constructor(private authService: AuthService, private router: Router) {}

  ngOnInit() {
    if (localStorage.getItem('user') == null) {
      alert(' FORBIDDEN ACCESS  !!');
      this.router.navigate(['/']);
    }

    const userString = localStorage.getItem('user');
    if (userString !== null) {
      this.user = JSON.parse(userString);
    }
    this.authService.getUser(this.user.username).subscribe(
      (data) => {
        this.user = data;
      },

      (err) => console.log(err)
    );
  }

  updateUser() {
    let updatedUser: any = {
      cin: this.user.cin,

      dateBirth: this.user.dateBirth,
      name: this.user.name,
      surname: this.user.surname,
      email: this.user.email,
      username: this.user.username,
      password: this.user.password,
      tel: this.user.tel,
    };

    console.log(updatedUser);

    this.authService.update(this.user.username, updatedUser).subscribe(
      (data) => {
        console.log(data);
        localStorage.clear();
        this.authService.getUser(this.user.username).subscribe(
          (data) => {
            this.userByUsername = data;

            const stockage = JSON.stringify(this.userByUsername);
            localStorage.setItem('user', stockage);
          },
          (err) => {
            console.log(err);
          }
        );
      },

      (err) => {
        console.log(err);
      }
    );
  }
}
